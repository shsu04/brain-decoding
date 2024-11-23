from .gwilliams import GWilliams

# from .schoffelen import Schoffelen
from .study import Study

STUDIES = {
    "gwilliams": GWilliams,
    # "schoffelen": Schoffelen,
}


class StudyFactory:
    @classmethod
    def create_study(cls, study_name: str, path: str) -> Study:
        if study_name not in STUDIES:
            raise ValueError(f"Study {study_name} not found")
        return STUDIES[study_name](path)
