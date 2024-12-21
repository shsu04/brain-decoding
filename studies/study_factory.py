from .gwilliams2023 import Gwilliams2023
from .schoffelen2022 import Schoffelen2022
from .study import Study

STUDIES = {
    "gwilliams2023": Gwilliams2023,
    "schoffelen2022": Schoffelen2022,
}


class StudyFactory:
    @classmethod
    def create_study(
        cls,
        study_name: str,
        batch_type: str,
        path: str,
        cache_enabled: bool,
        max_cache_size: int,
        cache_name: str,
    ) -> Study:
        if study_name not in STUDIES:
            raise ValueError(f"Study {study_name} not found")
        return STUDIES[study_name](
            path=path,
            batch_type=batch_type,
            cache_enabled=cache_enabled,
            max_cache_size=max_cache_size,
            cache_name=cache_name,
        )
