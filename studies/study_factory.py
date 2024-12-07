from .gwilliams2023 import GWilliams2023
from .schoffelen2022 import Schoffelen2022
from .study import Study

STUDIES = {
    "gwilliams": GWilliams2023,
    "schoffelen": Schoffelen2022,
}


class StudyFactory:
    @classmethod
    def create_study(cls, study_name: str, path: str) -> Study:
        if study_name not in STUDIES:
            raise ValueError(f"Study {study_name} not found")
        return STUDIES[study_name](path)
