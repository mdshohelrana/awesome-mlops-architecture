import bentoml
from bentoml import env, artifacts, api

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([
    bentoml.artifact.ModelArtifact('model1'),
    bentoml.artifact.ModelArtifact('model2')
])
class MyService(bentoml.BentoService):
    @api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
    def predict_model1(self, json_input):
        model = self.artifacts.model1
        return model.predict(json_input)

    @api(input=bentoml.io.JSON(), output=bentoml.io.JSON())
    def predict_model2(self, json_input):
        model = self.artifacts.model2
        return model.predict(json_input)