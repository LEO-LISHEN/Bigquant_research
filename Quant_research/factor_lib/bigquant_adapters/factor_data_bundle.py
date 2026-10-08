# -*- coding: utf-8 -*-
"""BigQuant 因子原始数据的分粒度容器。

本模块不查询数据，也不计算因子。它只负责把不同主键粒度的
DataFrame 分开保存，并提供统一验证、按日期裁剪和读取接口。

性能说明
--------
容器在创建时为每个包含 ``date`` 的数据域一次性建立日期索引：

1. 日期已经连续排序时，缓存每个日期对应的行切片边界；
2. 日期未排序时，缓存每个日期对应的整数行位置；
3. ``missing_dates`` 直接读取缓存，不再扫描整个 DataFrame；
4. ``select_dates`` 直接定位所需日期，不再对完整面板反复执行
   ``panel["date"].isin(...)``。

容器公开接口保持不变。为保证缓存有效，构造容器后应把其中的
DataFrame 视为只读对象；如需替换数据域，请使用 ``with_domain``。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFAULT_DOMAIN_KEYS = {
    "security_daily": ("date", "instrument"),
    "market_daily": ("date", "market_index"),
}


class FactorDataBundle:
    """保存因子计算所需的多个原始数据域。

    容器内部不跨粒度合并数据。例如 ``market_daily`` 不会被广播到
    ``security_daily`` 的每只股票行上。
    """

    def __init__(
        self,
        domains,
        key_columns=None,
        dependencies=None,
        dependency_target_dates=None,
        dependency_raw_dates=None,
    ):
        if not isinstance(domains, Mapping):
            raise TypeError("domains 必须是数据域名称到DataFrame的映射。")
        if not domains:
            raise ValueError("domains 不能为空。")

        custom_keys = {} if key_columns is None else dict(key_columns)
        normalized_domains = {}
        normalized_keys = {}

        for domain_name, panel in domains.items():
            name = self._normalize_domain_name(domain_name)
            keys = custom_keys.get(name, DEFAULT_DOMAIN_KEYS.get(name))
            if keys is None:
                keys = self._infer_key_columns(panel, name)
            keys = tuple(keys)
            normalized_domains[name] = self._validate_panel(
                panel,
                name,
                keys,
            )
            normalized_keys[name] = keys

        unknown_key_domains = sorted(
            set(custom_keys) - set(normalized_domains)
        )
        if unknown_key_domains:
            raise ValueError(
                "key_columns包含未提供的数据域："
                f"{unknown_key_domains}。"
            )

        self._domains = normalized_domains
        self._key_columns = normalized_keys
        self._dependencies = self._validate_dependencies(dependencies)
        self._dependency_target_dates = self._validate_dependency_date_maps(
            dependency_target_dates,
            "dependency_target_dates",
        )
        self._dependency_raw_dates = self._validate_dependency_date_maps(
            dependency_raw_dates,
            "dependency_raw_dates",
        )
        self._validate_dependency_metadata()
        self._build_date_indexes()

    @classmethod
    def _from_validated_domains(
        cls,
        domains,
        key_columns,
        dependencies=None,
        dependency_target_dates=None,
        dependency_raw_dates=None,
    ):
        """由已验证面板建立容器，避免日期裁剪后重复做全量校验。

        本方法仅供 ``select_dates`` 内部使用。裁剪结果来自已经通过
        主键、日期和重复值检查的数据域，因此无需再次执行相同校验。
        """
        instance = cls.__new__(cls)
        instance._domains = dict(domains)
        instance._key_columns = dict(key_columns)
        instance._dependencies = dict(dependencies or {})
        instance._dependency_target_dates = dict(
            dependency_target_dates or {}
        )
        instance._dependency_raw_dates = dict(
            dependency_raw_dates or {}
        )
        instance._build_date_indexes()
        return instance

    @classmethod
    def _validate_dependencies(cls, dependencies):
        if dependencies is None:
            return {}
        if not isinstance(dependencies, Mapping):
            raise TypeError("dependencies 必须是名称到 FactorDataBundle 的映射。")

        normalized = {}
        for dependency_name, bundle in dependencies.items():
            name = cls._normalize_domain_name(dependency_name)
            if not isinstance(bundle, cls):
                raise TypeError(
                    f"依赖 {name!r} 必须是 FactorDataBundle，"
                    f"实际为 {type(bundle).__name__}。"
                )
            normalized[name] = bundle
        return normalized

    @classmethod
    def _validate_dependency_date_maps(cls, date_maps, field_name):
        if date_maps is None:
            return {}
        if not isinstance(date_maps, Mapping):
            raise TypeError(f"{field_name} 必须是字典。")

        normalized = {}
        for dependency_name, per_target in date_maps.items():
            name = cls._normalize_domain_name(dependency_name)
            if not isinstance(per_target, Mapping):
                raise TypeError(
                    f"{field_name}[{name!r}] 必须是目标日到日期序列的字典。"
                )
            normalized_per_target = {}
            for target_date, values in per_target.items():
                target = pd.Timestamp(target_date).normalize()
                normalized_per_target[target] = cls._normalize_dates(
                    values,
                    allow_empty=False,
                )
            normalized[name] = normalized_per_target
        return normalized

    def _validate_dependency_metadata(self):
        dependency_names = set(self._dependencies)
        target_names = set(self._dependency_target_dates)
        raw_names = set(self._dependency_raw_dates)
        if dependency_names != target_names or dependency_names != raw_names:
            raise ValueError(
                "dependencies、dependency_target_dates 与 dependency_raw_dates "
                "必须包含完全相同的依赖名称。"
            )

        for dependency_name in dependency_names:
            target_keys = set(
                self._dependency_target_dates[dependency_name]
            )
            raw_keys = set(self._dependency_raw_dates[dependency_name])
            if target_keys != raw_keys:
                raise ValueError(
                    f"依赖 {dependency_name!r} 的目标日期映射与原始日期映射"
                    "键集合不一致。"
                )

    @staticmethod
    def _normalize_domain_name(domain_name):
        if not isinstance(domain_name, str) or not domain_name.strip():
            raise ValueError("数据域名称必须是非空字符串。")
        return domain_name.strip()

    @staticmethod
    def _infer_key_columns(panel, domain_name):
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(
                f"数据域 {domain_name!r} 必须是pandas.DataFrame。"
            )
        if {"date", "instrument"}.issubset(panel.columns):
            return ("date", "instrument")
        if {"date", "market_index"}.issubset(panel.columns):
            return ("date", "market_index")
        if "date" in panel.columns:
            return ("date",)
        raise ValueError(
            f"无法推断数据域 {domain_name!r} 的主键，请显式传入"
            "key_columns。"
        )

    @staticmethod
    def _validate_panel(panel, domain_name, key_columns):
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(
                f"数据域 {domain_name!r} 必须是pandas.DataFrame。"
            )
        if not key_columns:
            raise ValueError(
                f"数据域 {domain_name!r} 的主键不能为空。"
            )

        missing = sorted(set(key_columns) - set(panel.columns))
        if missing:
            raise ValueError(
                f"数据域 {domain_name!r} 缺少主键字段：{missing}。"
            )

        result = panel.copy(deep=False)
        if "date" in result.columns:
            parsed_dates = pd.to_datetime(
                result["date"],
                errors="coerce",
            ).dt.normalize()
            if parsed_dates.isna().any():
                raise ValueError(
                    f"数据域 {domain_name!r} 包含无效date。"
                )
            if not parsed_dates.equals(result["date"]):
                result = result.copy()
                result["date"] = parsed_dates

        for key in key_columns:
            if result[key].isna().any():
                raise ValueError(
                    f"数据域 {domain_name!r} 的主键 {key!r} 包含空值。"
                )

        duplicated = result.duplicated(list(key_columns), keep=False)
        if duplicated.any():
            examples = (
                result.loc[duplicated, list(key_columns)]
                .head(5)
                .astype(str)
                .to_dict("records")
            )
            raise ValueError(
                f"数据域 {domain_name!r} 在主键{list(key_columns)}上"
                f"存在重复记录：{examples}"
            )
        return result

    @staticmethod
    def _normalize_dates(dates, allow_empty=False):
        """将单日期或日期序列标准化为保持输入顺序的唯一日期索引。"""
        if isinstance(dates, (str, pd.Timestamp, np.datetime64)):
            dates = [dates]

        try:
            values = list(dates)
        except TypeError as exc:
            raise TypeError("dates必须是日期或日期序列。") from exc

        normalized = pd.DatetimeIndex(
            pd.to_datetime(values, errors="raise")
        ).normalize().unique()
        if not allow_empty and len(normalized) == 0:
            raise ValueError("dates不能为空。")
        return normalized

    def _build_date_indexes(self):
        """为所有日期型数据域建立一次性日期定位缓存。"""
        self._available_dates = {}
        self._date_indexes = {}

        for domain_name, panel in self._domains.items():
            if "date" not in panel.columns:
                continue

            if panel.empty:
                self._available_dates[domain_name] = pd.DatetimeIndex([])
                self._date_indexes[domain_name] = {
                    "mode": "empty",
                    "date_order": (),
                    "locations": {},
                }
                continue

            date_series = panel["date"]

            # BigQuant适配器通常按date排序。此时每个日期对应连续行块，
            # 只缓存(start, stop)即可，内存开销与交易日数量成正比。
            if date_series.is_monotonic_increasing:
                date_values = date_series.to_numpy(copy=False)
                change_points = (
                    np.flatnonzero(date_values[1:] != date_values[:-1]) + 1
                )
                starts = np.concatenate(
                    (np.array([0], dtype=np.int64), change_points)
                )
                stops = np.concatenate(
                    (change_points, np.array([len(panel)], dtype=np.int64))
                )

                date_order = tuple(
                    pd.Timestamp(date_values[start]).normalize()
                    for start in starts
                )
                locations = {
                    date: (int(start), int(stop))
                    for date, start, stop in zip(
                        date_order,
                        starts,
                        stops,
                    )
                }
                mode = "slices"

            else:
                # 自定义面板可能没有按日期排序。此时保存原始整数行位置，
                # select_dates仍按照原DataFrame行顺序返回结果。
                grouped_positions = date_series.groupby(
                    date_series,
                    sort=False,
                ).indices
                date_order = tuple(
                    pd.Timestamp(date).normalize()
                    for date in grouped_positions
                )
                locations = {
                    pd.Timestamp(date).normalize(): np.asarray(
                        positions,
                        dtype=np.int64,
                    )
                    for date, positions in grouped_positions.items()
                }
                mode = "positions"

            self._available_dates[domain_name] = pd.DatetimeIndex(
                date_order
            )
            self._date_indexes[domain_name] = {
                "mode": mode,
                "date_order": date_order,
                "locations": locations,
            }

    def _select_domain_dates(self, domain_name, selected_dates):
        """使用缓存直接裁剪一个数据域，并保持原始行顺序。"""
        panel = self._domains[domain_name]
        cache = self._date_indexes[domain_name]
        if cache["mode"] == "empty":
            return panel.iloc[0:0].copy()

        selected_set = set(selected_dates)
        ordered_dates = [
            date
            for date in cache["date_order"]
            if date in selected_set
        ]
        if not ordered_dates:
            return panel.iloc[0:0].copy()

        locations = cache["locations"]
        if cache["mode"] == "slices":
            bounds = [locations[date] for date in ordered_dates]

            # 连续交易日窗口是策略最常见的请求。连续时只做一次iloc切片，
            # 不创建数十万行的布尔掩码或整数位置数组。
            is_contiguous = all(
                previous_stop == current_start
                for (_, previous_stop), (current_start, _) in zip(
                    bounds,
                    bounds[1:],
                )
            )
            if is_contiguous:
                return panel.iloc[bounds[0][0]:bounds[-1][1]].copy()

            pieces = [
                panel.iloc[start:stop]
                for start, stop in bounds
            ]
            return pd.concat(
                pieces,
                axis=0,
                copy=False,
            ).copy()

        position_parts = [locations[date] for date in ordered_dates]
        positions = np.concatenate(position_parts)
        positions.sort()
        return panel.iloc[positions].copy()

    @property
    def domain_names(self):
        return tuple(self._domains)

    @property
    def dependency_names(self):
        return tuple(self._dependencies)

    def has_domain(self, domain_name):
        return domain_name in self._domains

    def get_domain(self, domain_name):
        if domain_name not in self._domains:
            raise KeyError(
                f"缺少数据域 {domain_name!r}；当前数据域："
                f"{list(self._domains)}。"
            )
        return self._domains[domain_name]

    def get_security_daily(self):
        return self.get_domain("security_daily")

    def has_dependency(self, dependency_name):
        return dependency_name in self._dependencies

    def get_dependency(self, dependency_name):
        if dependency_name not in self._dependencies:
            raise KeyError(
                f"缺少因子依赖 {dependency_name!r}；当前依赖："
                f"{list(self._dependencies)}。"
            )
        return self._dependencies[dependency_name]

    def get_dependency_target_dates(self, dependency_name, final_dates):
        """返回指定最终目标日对应的依赖因子计算截面日期。"""
        self.get_dependency(dependency_name)
        selected_final_dates = self._normalize_dates(final_dates)
        mapping = self._dependency_target_dates[dependency_name]
        missing = [date for date in selected_final_dates if date not in mapping]
        if missing:
            raise KeyError(
                f"依赖 {dependency_name!r} 缺少最终目标日映射："
                f"{[date.strftime('%Y-%m-%d') for date in missing]}。"
            )
        values = []
        for final_date in selected_final_dates:
            values.extend(mapping[final_date])
        return pd.DatetimeIndex(values).unique().sort_values()

    def key_columns(self, domain_name):
        if domain_name not in self._key_columns:
            raise KeyError(f"未知数据域：{domain_name!r}。")
        return self._key_columns[domain_name]

    def row_counts(self):
        return {
            domain_name: len(panel)
            for domain_name, panel in self._domains.items()
        }

    def missing_dates(self, domain_name, dates):
        """返回指定数据域缺少的日期，不再扫描完整面板。"""
        self.get_domain(domain_name)
        if domain_name not in self._available_dates:
            raise ValueError(
                f"数据域 {domain_name!r} 不包含date字段，无法检查日期。"
            )

        required_dates = self._normalize_dates(
            dates,
            allow_empty=True,
        )
        return required_dates.difference(
            self._available_dates[domain_name]
        )

    def select_dates(self, dates):
        """裁剪最终目标日，同时保留各依赖自身所需的预热日期。"""
        selected_dates = self._normalize_dates(dates)

        selected_domains = {}
        for domain_name, panel in self._domains.items():
            if "date" not in panel.columns:
                selected_domains[domain_name] = panel
                continue
            selected_domains[domain_name] = self._select_domain_dates(
                domain_name,
                selected_dates,
            )

        selected_dependencies = {}
        selected_dependency_targets = {}
        selected_dependency_raw = {}
        for dependency_name, bundle in self._dependencies.items():
            target_mapping = self._dependency_target_dates[dependency_name]
            raw_mapping = self._dependency_raw_dates[dependency_name]
            missing = [
                date
                for date in selected_dates
                if date not in target_mapping or date not in raw_mapping
            ]
            if missing:
                raise KeyError(
                    f"依赖 {dependency_name!r} 缺少最终目标日映射："
                    f"{[date.strftime('%Y-%m-%d') for date in missing]}。"
                )

            raw_dates = []
            for final_date in selected_dates:
                raw_dates.extend(raw_mapping[final_date])
            selected_dependencies[dependency_name] = bundle.select_dates(
                pd.DatetimeIndex(raw_dates).unique().sort_values()
            )
            selected_dependency_targets[dependency_name] = {
                final_date: target_mapping[final_date]
                for final_date in selected_dates
            }
            selected_dependency_raw[dependency_name] = {
                final_date: raw_mapping[final_date]
                for final_date in selected_dates
            }

        # 所有裁剪结果均来自已经验证的数据域，因此不重复执行日期解析、
        # 主键空值检查和重复值扫描；新容器只为裁剪结果建立轻量日期索引。
        return self._from_validated_domains(
            selected_domains,
            key_columns=self._key_columns,
            dependencies=selected_dependencies,
            dependency_target_dates=selected_dependency_targets,
            dependency_raw_dates=selected_dependency_raw,
        )

    def with_domain(self, domain_name, panel, key_columns=None):
        """返回替换或新增指定数据域后的新容器。"""
        name = self._normalize_domain_name(domain_name)
        domains = dict(self._domains)
        domains[name] = panel
        keys = dict(self._key_columns)
        if key_columns is not None:
            keys[name] = tuple(key_columns)
        elif name not in keys and name in DEFAULT_DOMAIN_KEYS:
            keys[name] = DEFAULT_DOMAIN_KEYS[name]
        return FactorDataBundle(
            domains,
            key_columns=keys,
            dependencies=self._dependencies,
            dependency_target_dates=self._dependency_target_dates,
            dependency_raw_dates=self._dependency_raw_dates,
        )

    def as_dict(self):
        """返回浅复制的数据域字典；DataFrame本身不会被复制。"""
        return dict(self._domains)

    def __contains__(self, domain_name):
        return self.has_domain(domain_name)

    def __repr__(self):
        details = ", ".join(
            f"{name}={len(panel):,} rows"
            for name, panel in self._domains.items()
        )
        if self._dependencies:
            details += ", dependencies=" + ",".join(self._dependencies)
        return f"FactorDataBundle({details})"
