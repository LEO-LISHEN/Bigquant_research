# -*- coding: utf-8 -*-
"""BigQuant 因子原始数据的分粒度容器。

本模块不查询数据，也不计算因子。它只负责把不同主键粒度的
DataFrame 分开保存，并提供统一验证、按日期裁剪和读取接口。
"""

from __future__ import annotations

from collections.abc import Mapping

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

    def __init__(self, domains, key_columns=None):
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

    @property
    def domain_names(self):
        return tuple(self._domains)

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
        panel = self.get_domain(domain_name)
        if isinstance(dates, (str, pd.Timestamp)):
            dates = [dates]
        required_dates = pd.DatetimeIndex(
            pd.to_datetime(list(dates), errors="raise")
        ).normalize().unique()
        available_dates = pd.DatetimeIndex(
            panel["date"].dropna().unique()
        ).normalize()
        return required_dates.difference(available_dates)

    def select_dates(self, dates):
        """对所有含date字段的数据域使用同一日期集合进行裁剪。"""
        if isinstance(dates, (str, pd.Timestamp)):
            dates = [dates]
        selected_dates = pd.DatetimeIndex(
            pd.to_datetime(list(dates), errors="raise")
        ).normalize().unique()
        if len(selected_dates) == 0:
            raise ValueError("dates不能为空。")

        selected_set = set(selected_dates)
        selected_domains = {}
        for domain_name, panel in self._domains.items():
            if "date" not in panel.columns:
                selected_domains[domain_name] = panel
                continue
            selected_domains[domain_name] = panel.loc[
                panel["date"].isin(selected_set)
            ].copy()

        return FactorDataBundle(
            selected_domains,
            key_columns=self._key_columns,
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
        return FactorDataBundle(domains, key_columns=keys)

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
        return f"FactorDataBundle({details})"
