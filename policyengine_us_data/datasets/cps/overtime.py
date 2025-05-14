from microimpute.models.qrf import QRFResults
from policyengine_us_data.datasets.cps.org import CensusCPSOrg
from policyengine_us_data.storage import STORAGE_FOLDER
import pickle


def train_exemption_status_model() -> QRFResults:

    org_df = CensusCPSOrg().load("main")

    # Add exemption status using rules

    ...

    # Train the model

    return ...


def get_tip_model() -> QRFResults:
    model_path = STORAGE_FOLDER / "tips.pkl"

    if not model_path.exists():
        model = train_exemption_status_model()

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    return model
