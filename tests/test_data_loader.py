import pytest
from pyspark.sql import SparkSession
from src.etl.data_loader import DataLoader

@pytest.fixture(scope="session")
def spark_fixture():
    # Reuse singleton SparkSession for tests
    spark = SparkSession.builder \
        .appName("TestDataLoader") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture
def sample_csv(tmp_path):
    # Create a small CSV file: 3 header lines + 2 data rows
    content = (
        "Title line to skip\n"
        "Header1,Header2,Header3,Header4,Header5\n"
        "SubH1,SubH2,SubH3,SubH4,SubH5\n"
        "2025-06-30T00:00+02:00,100.0,200.0,300.0,50.0\n"
        "2025-06-30T01:00+02:00,110.0,210.0,310.0,60.0\n"
    )
    file_path = tmp_path / "test.csv"
    file_path.write_text(content)
    return str(file_path)

def test_load_parses_rows(spark_fixture, sample_csv):
    # Given a CSV with two valid data lines
    loader = DataLoader(sample_csv)
    df = loader.load()
    rows = df.collect()

    # Expect two rows
    assert len(rows) == 2

    # Validate first row fields
    first = rows[0]
    assert first['timestamp'] == '2025-06-30T00:00+02:00'
    assert pytest.approx(first['nuclear'], rel=1e-3) == 100.0
    assert pytest.approx(first['nonrenewable'], rel=1e-3) == 200.0
    assert pytest.approx(first['renewable'], rel=1e-3) == 300.0
    assert pytest.approx(first['price'], rel=1e-3) == 50.0

    # Validate second row fields
    second = rows[1]
    assert second['timestamp'] == '2025-06-30T01:00+02:00'
    assert pytest.approx(second['nuclear'], rel=1e-3) == 110.0
    assert pytest.approx(second['nonrenewable'], rel=1e-3) == 210.0
    assert pytest.approx(second['renewable'], rel=1e-3) == 310.0
    assert pytest.approx(second['price'], rel=1e-3) == 60.0
