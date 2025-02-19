import uuid

import pytest

from griptape.artifacts.text_artifact import TextArtifact
from griptape.drivers.vector.griptape_cloud import GriptapeCloudVectorStoreDriver


class TestGriptapeCloudVectorStoreDriver:
    test_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    test_vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    test_namespaces = [str(uuid.uuid4()), str(uuid.uuid4())]
    test_metas = [{"key": "value1"}, {"key": "value2"}]
    test_scores = [0.7, 0.8]

    @pytest.fixture()
    def driver(self, mocker):
        test_entries = {
            "entries": [
                {
                    "id": self.test_ids[0],
                    "vector": self.test_vecs[0],
                    "namespace": self.test_namespaces[0],
                    "meta": self.test_metas[0],
                    "score": self.test_scores[0],
                },
                {
                    "id": self.test_ids[1],
                    "vector": self.test_vecs[1],
                    "namespace": self.test_namespaces[1],
                    "meta": self.test_metas[1],
                    "score": self.test_scores[1],
                },
            ]
        }

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = test_entries
        mocker.patch("requests.post", return_value=mock_response)

        return GriptapeCloudVectorStoreDriver(api_key="foo bar", knowledge_base_id="1")

    def test_query_vector(self, driver):
        with pytest.raises(NotImplementedError):
            driver.query_vector([0.0, 0.5])

    def test_query(self, driver):
        result = driver.query(
            "some query", count=10, namespace="foo", include_vectors=True, distance_metric="bar", filter={"foo": "bar"}
        )

        assert result[0].id == self.test_ids[0]
        assert result[1].id == self.test_ids[1]
        assert result[0].vector == self.test_vecs[0]
        assert result[1].vector == self.test_vecs[1]
        assert result[0].namespace == self.test_namespaces[0]
        assert result[1].namespace == self.test_namespaces[1]
        assert result[0].meta == self.test_metas[0]
        assert result[1].meta == self.test_metas[1]
        assert result[0].score == self.test_scores[0]
        assert result[1].score == self.test_scores[1]

    def test_query_defaults(self, driver):
        result = driver.query("some query")

        assert result[0].id == self.test_ids[0]
        assert result[1].id == self.test_ids[1]
        assert result[0].vector == self.test_vecs[0]
        assert result[1].vector == self.test_vecs[1]
        assert result[0].namespace == self.test_namespaces[0]
        assert result[1].namespace == self.test_namespaces[1]
        assert result[0].meta == self.test_metas[0]
        assert result[1].meta == self.test_metas[1]
        assert result[0].score == self.test_scores[0]
        assert result[1].score == self.test_scores[1]

    def test_query_artifact(self, driver):
        with pytest.raises(
            ValueError, match="GriptapeCloudVectorStoreDriver does not support querying with Artifacts."
        ):
            driver.query(TextArtifact("some query"))
