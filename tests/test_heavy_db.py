"""Heavy database-layer tests: introspection of hostile schemas, value-index
probing edge cases, evaluator result-equivalence semantics, and read-only
executor regressions."""
from __future__ import annotations

from pathlib import Path

import pytest

from nl2sql.db.introspect import SchemaBundle, introspect_database
from nl2sql.db.value_index import ValueIndex
from nl2sql.evaluation.evaluator import (
    NL2SQLEvaluator,
    execute_sqlite_readonly,
    results_equivalent,
)
from nl2sql.datasets.prepare import ExampleRecord

from conftest import SHOP_DDL, create_sqlite


WEIRD_DDL = """
CREATE TABLE "order" (id INTEGER PRIMARY KEY, qty INTEGER);
CREATE TABLE "weird table" (id INTEGER PRIMARY KEY, "txt col" TEXT);
CREATE TABLE "客户" (id INTEGER PRIMARY KEY, "名字" TEXT);
CREATE TABLE MixedCase (Id INTEGER PRIMARY KEY, Name TEXT);
CREATE TABLE cpk (a INTEGER, b INTEGER, PRIMARY KEY (a, b));
CREATE TABLE empty_t (id INTEGER);
CREATE TABLE employees (id INTEGER PRIMARY KEY, manager_id INTEGER REFERENCES employees(id));
CREATE VIEW v_customers AS SELECT id FROM "order";
INSERT INTO "order" VALUES (1, 5), (2, 7);
INSERT INTO "weird table" VALUES (1, 'hello'), (2, 'world');
INSERT INTO "客户" VALUES (1, '张三'), (2, '李四');
INSERT INTO MixedCase VALUES (1, 'Ann'), (2, 'Bo');
INSERT INTO employees VALUES (1, NULL), (2, 1), (3, 1);
"""


@pytest.fixture(scope="module")
def weird_db(tmp_path_factory: Path) -> Path:
    return create_sqlite(tmp_path_factory.mktemp("weird") / "weird.sqlite", WEIRD_DDL)


@pytest.fixture(scope="module")
def weird_bundle(weird_db: Path) -> SchemaBundle:
    return introspect_database(f"sqlite:///{weird_db.resolve()}", db_id="weird")


class TestIntrospectionHardening:
    def test_all_weird_tables_discovered(self, weird_bundle: SchemaBundle) -> None:
        names = {t.name for t in weird_bundle.tables}
        assert {"order", "weird table", "客户", "MixedCase", "cpk", "empty_t", "employees"} <= names

    def test_views_are_excluded(self, weird_bundle: SchemaBundle) -> None:
        names = {t.name for t in weird_bundle.tables}
        assert "v_customers" not in names

    @pytest.mark.parametrize(
        "query_name",
        ["order", "ORDER", "weird table", "WEIRD TABLE", "客户", "MIXEDCASE", "employees"],
    )
    def test_case_insensitive_lookup(self, weird_bundle: SchemaBundle, query_name: str) -> None:
        assert weird_bundle.get_table(query_name) is not None

    def test_composite_pk_members_marked_with_pk_tag(self, weird_bundle: SchemaBundle) -> None:
        cpk = weird_bundle.get_table("cpk")
        assert cpk is not None
        # Both composite-PK members carry the [PK] tag; each is NOT individually
        # unique (no phantom 1:1 cardinality downstream).
        assert all(c.is_pk for c in cpk.columns)
        assert not any(c.is_unique for c in cpk.columns)

    def test_empty_table_has_no_samples(self, weird_bundle: SchemaBundle) -> None:
        empty = weird_bundle.get_table("empty_t")
        assert empty is not None and empty.row_count == 0
        assert all(not c.sample_values for c in empty.columns)

    def test_self_referential_fk_recorded(self, weird_bundle: SchemaBundle) -> None:
        self_edges = [e for e in weird_bundle.join_edges if e.left_table == "employees" and e.right_table == "employees"]
        assert len(self_edges) == 1

    def test_unicode_roundtrip_save_load(self, weird_bundle: SchemaBundle, tmp_path: Path) -> None:
        p = weird_bundle.save(tmp_path / "bundle.json")
        loaded = SchemaBundle.load(p)
        assert {t.name for t in loaded.tables} == {t.name for t in weird_bundle.tables}
        assert loaded.ddl_hash == weird_bundle.ddl_hash

    def test_column_meanings_exact_and_lower_keys(self, weird_db: Path) -> None:
        meanings = {
            "weird|MixedCase|Name": "display name",
            "weird|order|qty": "quantity ordered",
        }
        bundle = introspect_database(f"sqlite:///{weird_db.resolve()}", db_id="weird", column_meanings=meanings)
        assert bundle.get_table("MixedCase").get_column("Name").business_description == "display name"
        assert bundle.get_table("order").get_column("qty").business_description == "quantity ordered"

    def test_sqlglot_compiles_quoted_identifiers(self, weird_bundle: SchemaBundle) -> None:
        from nl2sql.compiler.sql_compiler import compile_and_validate_sql

        for sql in ['SELECT * FROM "order"', 'SELECT * FROM "weird table"', "SELECT * FROM MixedCase"]:
            res = compile_and_validate_sql(sql, weird_bundle)
            assert res.used_tables, f"no tables resolved for {sql}"

    def test_ddl_hash_changes_with_schema(self, weird_db: Path, tmp_path: Path) -> None:
        url = f"sqlite:///{weird_db.resolve()}"
        b1 = introspect_database(url, db_id="weird")
        other = create_sqlite(tmp_path / "other.sqlite", SHOP_DDL)
        b2 = introspect_database(f"sqlite:///{other.resolve()}", db_id="other")
        assert b1.ddl_hash != b2.ddl_hash

    def test_row_counts_accurate(self, weird_bundle: SchemaBundle) -> None:
        by_name = {t.name: t for t in weird_bundle.tables}
        assert by_name["order"].row_count == 2
        assert by_name["empty_t"].row_count == 0
        assert by_name["employees"].row_count == 3


class TestValueIndexHardening:
    @pytest.fixture(scope="class")
    def shop_idx(self, tmp_path_factory: Path) -> ValueIndex:
        tmp = tmp_path_factory.mktemp("vi")
        db = create_sqlite(tmp / "shop.sqlite", SHOP_DDL)
        bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="shop")
        idx = ValueIndex(tmp / "vals.sqlite")
        idx.build_from_db(f"sqlite:///{db.resolve()}", bundle)
        return idx

    def test_long_question_still_matches_late_unigram(self, shop_idx: ValueIndex) -> None:
        """Regression: 4/3/2-grams used to fill the whole phrase budget so late
        unigrams (customer names) silently stopped matching on long questions."""
        q = (
            "please tell me about the support tickets that were filed by the "
            "customer named Alice during the last fiscal quarter of the year"
        )
        hits = shop_idx.probe(q)
        values = {h.value for h in hits}
        assert "Alice" in values

    def test_short_question_matches(self, shop_idx: ValueIndex) -> None:
        hits = shop_idx.probe("Alice")
        assert any(h.value == "Alice" for h in hits)

    def test_exact_match_outranks_partial(self, shop_idx: ValueIndex) -> None:
        hits = shop_idx.probe("open tickets Alice")
        if len(hits) >= 2:
            assert hits[0].score >= hits[-1].score

    def test_top_k_respected(self, shop_idx: ValueIndex) -> None:
        assert len(shop_idx.probe("open closed Alice Bob CA NY", top_k=3)) <= 3

    def test_fts_special_characters_do_not_crash(self, shop_idx: ValueIndex) -> None:
        for q in [
            'find "quoted" AND OR NEAR(x) things',
            "what about -minus ^caret *star stuff",
            "a" * 500,
            "!!! ??? ###",
            "",
        ]:
            assert isinstance(shop_idx.probe(q), list)

    def test_empty_index_probes_return_nothing(self, tmp_path: Path) -> None:
        idx = ValueIndex(tmp_path / "empty.sqlite")
        assert idx.probe("anything Alice") == []

    def test_rebuild_on_schema_change(self, tmp_path: Path) -> None:
        tmp = tmp_path
        db1 = create_sqlite(tmp / "v1.sqlite", SHOP_DDL)
        b1 = introspect_database(f"sqlite:///{db1.resolve()}", db_id="v1")
        idx = ValueIndex(tmp / "idx.sqlite")
        n1 = idx.build_from_db(f"sqlite:///{db1.resolve()}", b1)
        assert n1 > 0
        assert idx.is_built_for(b1.ddl_hash)

        db2 = create_sqlite(tmp / "v2.sqlite", WEIRD_DDL)
        b2 = introspect_database(f"sqlite:///{db2.resolve()}", db_id="v2")
        assert not idx.is_built_for(b2.ddl_hash)
        n2 = idx.build_from_db(f"sqlite:///{db2.resolve()}", b2)
        assert n2 > 0
        assert idx.is_built_for(b2.ddl_hash)
        assert not idx.is_built_for(b1.ddl_hash)

    def test_rebuild_is_idempotent(self, tmp_path: Path) -> None:
        db = create_sqlite(tmp_path / "s.sqlite", SHOP_DDL)
        bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="s")
        idx = ValueIndex(tmp_path / "i.sqlite")
        idx.build_from_db(f"sqlite:///{db.resolve()}", bundle)
        assert idx.build_from_db(f"sqlite:///{db.resolve()}", bundle) == 0  # already built

    def test_numeric_columns_indexed_only_below_cardinality_cutoff(self, tmp_path: Path) -> None:
        # 150 distinct amounts: above the low-cardinality cutoff, never indexed.
        ddl = (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, amount REAL);\n"
            + "".join(f"INSERT INTO t VALUES ({i}, {i + 0.5});" for i in range(1, 151))
        )
        db = create_sqlite(tmp_path / "hi.sqlite", ddl)
        bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="hi")
        idx = ValueIndex(tmp_path / "hi_vi.sqlite")
        idx.build_from_db(f"sqlite:///{db.resolve()}", bundle, low_cardinality_max_distinct=100)
        hits = idx.probe("amounts 1.5 2.5")
        assert all(h.column_name != "amount" for h in hits)

        # SHOP_DDL amounts (3 distinct values <= 100): indexed for grounding.
        db2 = create_sqlite(tmp_path / "lo.sqlite", SHOP_DDL)
        bundle2 = introspect_database(f"sqlite:///{db2.resolve()}", db_id="lo")
        idx2 = ValueIndex(tmp_path / "lo_vi.sqlite")
        idx2.build_from_db(f"sqlite:///{db2.resolve()}", bundle2, low_cardinality_max_distinct=100)
        lo_hits = idx2.probe("orders with amount 120.5")
        assert any(h.column_name == "amount" and h.value == "120.5" for h in lo_hits)

    def test_overlong_values_skipped(self, tmp_path: Path) -> None:
        ddl = (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, txt TEXT);"
            "INSERT INTO t VALUES (1, '" + "x" * 300 + "'), (2, 'short');"
        )
        db = create_sqlite(tmp_path / "long.sqlite", ddl)
        bundle = introspect_database(f"sqlite:///{db.resolve()}", db_id="long")
        idx = ValueIndex(tmp_path / "li.sqlite")
        idx.build_from_db(f"sqlite:///{db.resolve()}", bundle)
        hits = idx.probe("short")
        assert any(h.value == "short" for h in hits)
        assert all(len(h.value) <= 120 for h in hits)


class TestResultsEquivalence:
    def test_int_and_float_rows_are_equivalent(self) -> None:
        assert results_equivalent("SELECT 1", [(1,)], [(1.0,)])
        assert results_equivalent("SELECT 1", [(5, 3)], [(5.0, 3.0)])

    def test_float_precision_boundary(self) -> None:
        assert results_equivalent("SELECT 1", [(120.5,)], [(120.50001,)])
        assert not results_equivalent("SELECT 1", [(120.5,)], [(120.6,)])

    def test_null_and_empty_string_distinct(self) -> None:
        assert not results_equivalent("SELECT 1", [(None,)], [("",)])
        assert results_equivalent("SELECT 1", [(None,)], [(None,)])

    def test_both_empty_equivalent(self) -> None:
        assert results_equivalent("SELECT 1", [], [])

    def test_unordered_rows_equivalent_regardless_of_order(self) -> None:
        g = [("Alice", 1), ("Bob", 2)]
        p = [("Bob", 2), ("Alice", 1)]
        assert results_equivalent("SELECT name, id FROM customers", g, p)

    def test_toplevel_order_by_enforced(self) -> None:
        g = [("Alice",), ("Bob",)]
        p = [("Bob",), ("Alice",)]
        assert not results_equivalent("SELECT name FROM customers ORDER BY name", g, p)

    def test_subquery_order_by_does_not_force_strict_order(self) -> None:
        """ORDER BY inside a subquery does not define final row order."""
        gold = "SELECT id FROM (SELECT id, name FROM customers ORDER BY name)"
        g = [(1,), (2,)]
        p = [(2,), (1,)]
        assert results_equivalent(gold, g, p)

    def test_whitespace_strings_stripped(self) -> None:
        assert results_equivalent("SELECT 1", [("  Alice ",)], [("Alice",)])


class TestEvaluatorMetricsAccounting:
    def _record(self, rid: str, sql: str, shop_db: Path) -> ExampleRecord:
        return ExampleRecord(
            id=rid, source="t", db_id="shop", db_path=shop_db, question=f"q {rid}", sql=sql,
            gold_tables=["tickets"], gold_columns=["tickets.id"],
        )

    def test_abstention_and_auroc(self, shop_db: Path) -> None:
        records = [
            self._record("r1", "SELECT COUNT(*) FROM tickets", shop_db),
            self._record("r2", "SELECT COUNT(*) FROM customers", shop_db),
            self._record("r3", "SELECT COUNT(*) FROM orders", shop_db),
        ]
        preds = {
            "r1": {"sql": "SELECT COUNT(*) FROM tickets", "abstained": False, "confidence": 0.95, "linked_tables": ["tickets"]},
            "r2": {"sql": None, "abstained": True, "confidence": 0.30, "linked_tables": []},
            "r3": {"sql": "SELECT COUNT(*) FROM tickets", "abstained": False, "confidence": 0.40, "linked_tables": ["tickets"]},
        }
        metrics = NL2SQLEvaluator().evaluate(records, lambda r: preds[r.id], dataset_name="heavy")
        assert metrics.total_examples == 3
        assert metrics.abstention_rate == round(1 / 3, 4)
        assert metrics.executable_rate == round(2 / 3, 4)
        assert metrics.execution_accuracy == round(1 / 3, 4)  # only r1 matches gold
        assert metrics.selective_execution_accuracy == round(1 / 2, 4)
        assert metrics.verifier_auroc == 1.0  # 0.95 correct, 0.40 wrong, 0.30 abstained

    def test_latency_and_throughput_sane(self, shop_db: Path) -> None:
        records = [self._record(f"l{i}", "SELECT COUNT(*) FROM tickets", shop_db) for i in range(5)]
        metrics = NL2SQLEvaluator().evaluate(
            records,
            lambda r: {"sql": r.sql, "abstained": False, "confidence": 0.9, "linked_tables": ["tickets"]},
            dataset_name="lat",
        )
        assert metrics.p50_latency_ms >= 0
        assert metrics.p95_latency_ms >= metrics.p50_latency_ms
        assert metrics.throughput_qps > 0


class TestExecutorRegressions:
    def test_query_only_does_not_break_selects(self, shop_db: Path) -> None:
        ok, rows, _, _ = execute_sqlite_readonly(shop_db, "SELECT COUNT(*) FROM customers")
        assert ok and rows == [(2,)]

    def test_with_prefix_select_allowed(self, shop_db: Path) -> None:
        ok, rows, _, _ = execute_sqlite_readonly(shop_db, "WITH c AS (SELECT 1 AS v) SELECT v FROM c")
        assert ok and rows == [(1,)]

    def test_comment_prefixed_select_allowed(self, shop_db: Path) -> None:
        ok, rows, _, _ = execute_sqlite_readonly(shop_db, "/* header */ SELECT COUNT(*) FROM customers")
        assert ok and rows == [(2,)]
