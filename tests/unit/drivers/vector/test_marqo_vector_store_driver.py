from collections import namedtuple

import pytest

from griptape.artifacts import TextArtifact
from griptape.drivers.vector.marqo import MarqoVectorStoreDriver
from tests.mocks.mock_embedding_driver import MockEmbeddingDriver


class TestMarqoVectorStorageDriver:
    @pytest.fixture(autouse=True)
    def mock_marqo(self, mocker):
        # Create a fake responses
        fake_add_document_response = {
            "errors": False,
            "items": [{"_id": "5aed93eb-3878-4f12-bc92-0fda01c7d23d", "result": "created", "status": 201}],
            "processingTimeMs": 6,
            "index_name": "my-first-index",
        }

        fake_search_response = {
            "hits": [
                {
                    "Title": "Test Title",
                    "Description": "Test description",
                    "_highlights": {"Description": "Test highlight"},
                    "_id": "5aed93eb-3878-4f12-bc92-0fda01c7d23d",
                    "_source": {
                        "values": [0.1, 0.2, 0.3],
                        "metadata": {"Title": "Test title", "Description": "Test description"},
                    },
                    "_score": 0.6047464,
                }
            ],
            "limit": 10,
            "offset": 0,
            "processingTimeMs": 49,
            "query": "Test query",
        }

        fake_get_document_response = {
            "Blurb": "Test description",
            "Title": "Test Title",
            "_id": "5aed93eb-3878-4f12-bc92-0fda01c7d23d",
            "_tensor_facets": [
                {"Title": "Test Title", "_embedding": [-0.10393160581588745, 0.0465407557785511, -0.01760256476700306]},
                {
                    "Blurb": "Test description",
                    "_embedding": [-0.045681700110435486, 0.056278493255376816, 0.022254955023527145],
                },
            ],
        }

        fake_create_index_response = {"acknowledged": True, "shards_acknowledged": True, "index": "my-first-index"}

        # Define a namedtuple type for the mock indexes
        MockIndex = namedtuple("MockIndex", ["index_name"])

        # Create a list of mock indexes
        mock_indexes = [MockIndex(index_name="my-first-index"), MockIndex(index_name="test-index")]
        mock_get_indexes_response = {"results": mock_indexes}

        # Mock the marqo.Client
        mock_client = mocker.Mock()
        mocker.patch("marqo.Client", return_value=mock_client)

        # Mock the index method and the methods of the returned object
        mock_index = mocker.Mock()
        mock_client.index.return_value = mock_index
        mock_index.add_documents.return_value = fake_add_document_response
        mock_index.search.return_value = fake_search_response
        mock_index.get_document.return_value = fake_get_document_response

        # Mock the get_indexes method
        mock_client.get_indexes.return_value = mock_get_indexes_response

        # Mock the create_index method
        mock_client.create_index.return_value = fake_create_index_response

        # Return the mock_client for use in other fixtures
        return mock_client

    @pytest.fixture()
    def driver(self, mock_marqo):
        return MarqoVectorStoreDriver(
            api_key="foobar",
            url="http://localhost:8000",
            index="test",
            client=mock_marqo,
            embedding_driver=MockEmbeddingDriver(),
        )

    def test_upsert_text(self, driver, mock_marqo):
        result = driver.upsert_text("test text", vector_id="5aed93eb-3878-4f12-bc92-0fda01c7d23d")
        mock_marqo.index().add_documents.assert_called()
        assert result == "5aed93eb-3878-4f12-bc92-0fda01c7d23d"

    def test_upsert_text_artifact(self, driver, mock_marqo):
        # Arrange
        text = TextArtifact(id="a44b04ff052e4109b3c6fda0f3f3e997", value="racoons")
        mock_marqo.index().add_documents.return_value = {
            "errors": False,
            "items": [{"_id": text.id, "result": "created", "status": 201}],
            "processingTimeMs": 6,
            "index_name": "my-first-index",
        }

        # Act
        result = driver.upsert_text_artifact(text)

        # Check that the return value is as expected
        expected_return_value = {
            "errors": False,
            "items": [{"_id": text.id, "result": "created", "status": 201}],
            "processingTimeMs": 6,
            "index_name": "my-first-index",
        }
        assert result == expected_return_value["items"][0]["_id"]

    def test_query_vector(self, driver, mock_marqo):
        results = driver.query_vector([0.1, 0.2, 0.3])
        mock_marqo.index().search.assert_called()
        assert len(results) == 1
        assert results[0].score == 0.6047464
        assert results[0].meta["Title"] == "Test Title"
        assert results[0].meta["Description"] == "Test description"
        assert results[0].id == "5aed93eb-3878-4f12-bc92-0fda01c7d23d"

    def test_search(self, driver, mock_marqo):
        results = driver.query("Test query")
        mock_marqo.index().search.assert_called()
        assert len(results) == 1
        assert results[0].score == 0.6047464
        assert results[0].meta["Title"] == "Test Title"
        assert results[0].meta["Description"] == "Test description"
        assert results[0].id == "5aed93eb-3878-4f12-bc92-0fda01c7d23d"

    def test_search_with_include_vectors(self, driver, mock_marqo):
        # Act
        results = driver.query("Test query", include_vectors=True)

        # Assert
        mock_marqo.index().search.assert_called_once_with(
            "Test query", limit=5, attributes_to_retrieve=None, filter_string=None
        )
        mock_marqo.index().get_document.assert_called_once_with(
            "5aed93eb-3878-4f12-bc92-0fda01c7d23d", expose_facets=True
        )
        assert len(results) == 1
        assert results[0].score == 0.6047464
        assert results[0].meta["Title"] == "Test Title"
        assert results[0].meta["Description"] == "Test description"
        assert results[0].vector == [-0.10393160581588745, 0.0465407557785511, -0.01760256476700306]  # The vector
        # values should match the "_embedding" values of title in mock response

    def test_load_entry(self, driver, mock_marqo):
        # Mock 'get_document' method to return a dictionary
        entry = driver.load_entry("5aed93eb-3878-4f12-bc92-0fda01c7d23d")
        mock_marqo.index().get_document.assert_called_once_with(
            document_id="5aed93eb-3878-4f12-bc92-0fda01c7d23d", expose_facets=True
        )
        assert entry.id == "5aed93eb-3878-4f12-bc92-0fda01c7d23d"
        assert entry.meta["Title"] == "Test Title"
        assert entry.meta["Blurb"] == "Test description"
        assert entry.vector == [
            -0.10393160581588745,
            0.0465407557785511,
            -0.01760256476700306,
        ]  # The vector values should match the "_embedding" values of
        # title in mock response

    def test_load_entries(self, driver, mock_marqo):
        # Arrange
        fake_search_response = {
            "hits": [{"_id": "5aed93eb-3878-4f12-bc92-0fda01c7d23d"}],
            "limit": 10,
            "offset": 0,
            "processingTimeMs": 49,
            "query": "",
        }

        fake_get_documents_response = {
            "results": [
                {
                    "Title": "Test Title",
                    "Description": "Test description",
                    "_found": True,
                    "_id": "5aed93eb-3878-4f12-bc92-0fda01c7d23d",
                    "_tensor_facets": [{"_embedding": [0.1, 0.2, 0.3]}],
                }
            ]
        }

        mock_marqo.index().search.return_value = fake_search_response
        mock_marqo.index().get_documents.return_value = fake_get_documents_response

        # Act
        entries = driver.load_entries()

        # Assert
        assert len(entries) == 1
        mock_marqo.index().search.assert_called_once_with("", limit=10000)
        mock_marqo.index().get_documents.assert_called_once_with(
            document_ids=["5aed93eb-3878-4f12-bc92-0fda01c7d23d"], expose_facets=True
        )
        assert entries[0].id == "5aed93eb-3878-4f12-bc92-0fda01c7d23d"
        assert entries[0].vector == [0.1, 0.2, 0.3]
        assert entries[0].meta["Title"] == "Test Title"
        assert entries[0].meta["Description"] == "Test description"
