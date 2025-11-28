import pandas as pd
import numpy as np

import altair as alt
from jinja2 import Template

from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import os


class UtilsCalc:
    def __init__(self):
        pass

    def calculate_information_value(df, target):
        """
        Calculate the information value (IV) for each feature in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        target (str): The name of the target variable column.

        Returns:
        pd.DataFrame: A dataframe containing the information value for each feature.
        """

        iv = pd.DataFrame()
        df = df.dropna()
        for col in df.columns:
            epsilon = 1e-10
            if col != target:
                temp = df[[col, target]].copy()
                temp["total"] = 1
                if df[col].dtype == "object" or len(df[col].unique()) < 20:
                    # Binning for non-continuous variables
                    temp[col] = temp[col].astype(str)
                else:
                    # Binning for continuous variables
                    temp[col] = pd.qcut(temp[col], q=10, duplicates="drop")
                temp = (
                    temp.groupby([col, target], observed=True)
                    .count()
                    .unstack()
                    .fillna(0)
                )
                temp.columns = temp.columns.droplevel()
                temp["total"] = temp.sum(axis=1)
                temp["good"] = temp[0] / temp[0].sum()
                temp["bad"] = temp[1] / temp[1].sum()
                temp["woe"] = np.log(temp["good"] + epsilon / temp["bad"])
                temp["iv"] = (temp["good"] - temp["bad"]) * temp["woe"]
                iv[col] = [temp["iv"].sum()]

        iv_df = iv.T
        iv_df.columns = ["Information Value"]
        iv_df.replace([np.inf, -np.inf], 0, inplace=True)
        iv_df.sort_values(by="Information Value", ascending=False, inplace=True)

        return iv_df

    def calculate_mutual_information(df, target_column):
        """
        Calculate the mutual information between each feature and the target variable.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        target_column (str): The name of the target variable column.

        Returns:
        pd.DataFrame: A dataframe containing the mutual information values.
        """
        # Separate the features and the target variable

        df = df.dropna()
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Apply LabelEncoder to non-numeric columns
        for col in X.columns:
            if X[col].dtype not in ["float64", "int64"]:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        # Determine if the target variable is categorical or continuous
        if df[target_column].dtype == "object" or len(df[target_column].unique()) < 20:
            mi = mutual_info_classif(X, y)
        else:
            mi = mutual_info_regression(X, y)

        # Create a dataframe to store the mutual information values
        mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mi})
        mi_df.sort_values(by="Mutual Information", ascending=False, inplace=True)

        return mi_df

    def calculate_psi(df, time_column, variable_column, base_period, comparison_period):
        """
        Calculate the Index of Population Stability (psi) for a variable (categorical or continuous)
        between two time periods.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        time_column (str): The name of the column representing the time period.
        variable_column (str): The name of the variable column.
        base_period (str): The base time period for comparison.
        comparison_period (str): The time period to compare against the base period.

        Returns:
        float: The calculated psi value.
        """
        # Extract the data for the base and comparison periods
        base_data = df[df[time_column] == base_period][variable_column]
        comparison_data = df[df[time_column] == comparison_period][variable_column]

        # Check if the variable is categorical or continuous
        if base_data.dtype == "object" or len(base_data.unique()) < 20:
            # Calculate the distribution for categorical variables
            base_dist = base_data.value_counts(normalize=True)
            comparison_dist = comparison_data.value_counts(normalize=True)

            # Align the distributions to ensure they have the same categories
            base_dist, comparison_dist = base_dist.align(comparison_dist, fill_value=0)
        else:
            # Bin continuous variables into 10 equal-sized bins
            bins = pd.qcut(
                pd.concat([base_data, comparison_data]), q=10, duplicates="drop"
            )

            # Calculate the distribution for continuous variables
            base_dist = pd.cut(base_data, bins=bins.cat.categories).value_counts(
                normalize=True
            )
            comparison_dist = pd.cut(
                comparison_data, bins=bins.cat.categories
            ).value_counts(normalize=True)

        # Calculate the psi
        psi = sum(abs(base_dist - comparison_dist)) / 2

        return psi

    def calculate_ks(df, time_column, value_column, base_period, comparison_period):
        """
        Calculate the Kolmogorov-Smirnov (KS) statistic for a continuous variable between two time periods.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        time_column (str): The name of the column representing the time period.
        value_column (str): The name of the continuous variable column.
        base_period (str): The base time period for comparison.
        comparison_period (str): The time period to compare against the base period.

        Returns:
        float: The calculated KS statistic value.
        """

        # Extract the values for the base period and comparison period
        base_values = df[df[time_column] == base_period][value_column]
        comparison_values = df[df[time_column] == comparison_period][value_column]

        # Calculate the KS statistic
        ks_statistic, _ = ks_2samp(base_values, comparison_values)

        return ks_statistic

    def calculate_missing_zero(df):
        # Extract the number of missing values per variable
        missing_per_variable = df.isnull().sum().to_dict()

        # Calculate the percentage of missing values for each variable
        total_rows = df.shape[0]
        missing_percentage = {
            var: (count / total_rows) * 100
            for var, count in missing_per_variable.items()
        }

        # Count the number of variables with specific percentages of missing values
        missing_100 = sum(1 for perc in missing_percentage.values() if perc == 100)
        missing_75 = sum(1 for perc in missing_percentage.values() if perc >= 75)
        missing_50 = sum(1 for perc in missing_percentage.values() if perc >= 50)
        missing_25 = sum(1 for perc in missing_percentage.values() if perc >= 25)
        missing_0 = sum(1 for perc in missing_percentage.values() if perc == 0)

        # Extract the number of zero values per variable
        zero_per_variable = (df == 0).sum().to_dict()

        # Calculate the percentage of zero values for each variable
        zero_percentage = {
            var: (count / total_rows) * 100 for var, count in zero_per_variable.items()
        }

        # Count the number of variables with specific percentages of zero values
        zero_100 = sum(1 for perc in zero_percentage.values() if perc == 100)
        zero_75 = sum(1 for perc in zero_percentage.values() if perc >= 75)
        zero_50 = sum(1 for perc in zero_percentage.values() if perc >= 50)
        zero_25 = sum(1 for perc in zero_percentage.values() if perc >= 25)
        zero_0 = sum(1 for perc in zero_percentage.values() if perc == 0)

        # Extract the number of variables that are float or int and the number that are not
        num_float_int = sum(
            value
            for dtype, value in df.dtypes.value_counts().to_dict().items()
            if dtype in ["float32", "int32", "float64", "int64"]
        )
        num_not_float_int = df.shape[1] - num_float_int

        # Create a dataframe to store the results
        return {
            "description": [
                "missing 100% -> 0%",
                "zero 100% -> 0%",
                "number_of_continuous",
                "number_of_not_continuous",
            ],
            "value": [
                [missing_100, missing_75, missing_50, missing_25, missing_0],
                [zero_100, zero_75, zero_50, zero_25, zero_0],
                num_float_int,
                num_not_float_int,
            ],
        }


class SandEDA(UtilsCalc):
    def __init__(
        self,
        df: pd.DataFrame,
        target_name: str,
        time_name: str,
        id_name: str,
        top_n: int = 5,
    ):
        self.df = df.copy()
        self.target_name = target_name
        self.time_name = time_name
        self.id_name = id_name
        self.target_type = df[target_name].dtypes
        self.top_n = top_n

        for var in self.df.columns:
            if self.df[var].dtype == "object" and len(self.df[var].unique()) > 10:
                # Consolidate into top 30 categories and 'Other'
                top_categories = self.df[var].value_counts().nlargest(10).index
                self.df[var] = self.df[var].apply(
                    lambda x: x if (pd.isnull(x) or x in top_categories) else "Other"
                )

    def calc_general(self):
        res_general = {
            "dataset_general": {
                "number_of_rows": self.df.shape[0],
                "number_of_columns": self.df.shape[1],
                "number_of_cells": self.df.size,
                "number_of_missing": int(self.df.isnull().sum().sum()),
                "number_of_unique_types": int(self.df.dtypes.value_counts().count()),
                "number_of_zeros": int((self.df == 0).sum().sum()),
                "memory_usage_in_megabytes": float(
                    round(self.df.memory_usage().sum() / 1024**2, 2)
                ),
            },
            "variables_general": {
                "variables_names": self.df.columns.tolist(),
                "number_of_variables_per_type": self.df.dtypes.value_counts().to_dict(),
                "number_of_missing_per_variable": self.df.isnull().sum().to_dict(),
                "number_of_zero_per_variable": (self.df == 0).sum().to_dict(),
            },
            "target_general": {
                "target_name": self.target_name,
                "number_of_one": int((self.df[self.target_name] == 1).sum()),
                "number_of_unique_values": int(self.df[self.target_name].nunique()),
                "number_of_missing": int(self.df[self.target_name].isnull().sum()),
                "number_of_zero": int((self.df[self.target_name] == 0).sum()),
                "target_type": str(self.df[self.target_name].dtypes),
                "target_metric_time": {
                    time: float(
                        round(
                            (
                                self.df[self.df[self.time_name] == time][
                                    self.target_name
                                ].sum()
                                / self.df[self.df[self.time_name] == time].shape[0]
                            )
                            * 100,
                            2,
                        )
                    )
                    for time in self.df[self.time_name].unique()
                },
            },
            "missing_zero": UtilsCalc.calculate_missing_zero(self.df),
        }

        return res_general

    def histogram_variable_espec(self, variable):
        if len(variable.unique()) <= 50 or variable.dtype not in ["int64", "float64"]:
            res_hist_espec = dict(sorted(variable.value_counts().items()))
        else:
            res_hist_espec = dict(sorted(variable.value_counts(bins=50).items(), key=lambda x: x[0].left))
        return res_hist_espec

    def decil_variable_espec(self, variable_name):
        if len(self.df[variable_name].unique()) <= 10 or self.df[
            variable_name
        ].dtype not in ["int64", "float64"]:
            res_decil_espec = self.df.groupby([variable_name])[self.target_name].mean()
        else:
            res_decil_espec = self.df.groupby(
                pd.qcut(self.df[variable_name], q=10, labels=False, duplicates="drop")
            )[self.target_name].mean()
        return res_decil_espec

    def variables_espec(self):
        res_espec = {}

        df_filtered = self.df.drop(
            columns=[self.time_name, self.target_name, self.id_name]
        )

        for var in df_filtered:
            if self.df[var].dtypes == "object":
                res_espec[var] = {
                    "descriptive_statistics": {
                        "variable_type": self.df[var].dtypes,
                        "number_of_missing": int(self.df[var].isnull().sum()),
                        "number_of_unique_values": int(self.df[var].nunique()),
                        "number_per_values": self.df[var].value_counts().to_dict(),
                        "unique_values": self.df[var].unique().tolist(),
                    },
                    "quantile_statistics": {
                        "min": "cat feat",
                        "5%": "cat feat",
                        "25%": "cat feat",
                        "50%": "cat feat",
                        "75%": "cat feat",
                        "95%": "cat feat",
                        "max": "cat feat",
                    },
                    "histogram": self.histogram_variable_espec(self.df[var]),
                    "decil": self.decil_variable_espec(var),
                }
            elif self.df[var].dtype in ["int64", "float64"]:
                res_espec[var] = {
                    "descriptive_statistics": {
                        "variable_type": str(self.df[var].dtypes),
                        "number_of_unique_values": int(self.df[var].nunique()),
                        "number_of_missing": int(self.df[var].isnull().sum()),
                        "number_of_zeros": int((self.df[var] == 0).sum()),
                        "sum": float(self.df[var].sum()),
                        "mean": float(round(self.df[var].mean(), 2)),
                        "median": float(self.df[var].median()),
                        "std": float(round(self.df[var].std(), 2)),
                    },
                    "quantile_statistics": {
                        "min": float(self.df[var].min()),
                        "5%": float(self.df[var].quantile(0.05)),
                        "25%": float(self.df[var].quantile(0.25)),
                        "50%": float(self.df[var].quantile(0.5)),
                        "75%": float(self.df[var].quantile(0.75)),
                        "95%": float(self.df[var].quantile(0.95)),
                        "max": float(self.df[var].max()),
                    },
                    "histogram": self.histogram_variable_espec(self.df[var]),
                    "decil": self.decil_variable_espec(var),
                }

        return res_espec

    def variables_espec_time(self):
        res_espect_var = {}
        quintil_cut_points = {}

        df_filtered = self.df.drop(
            columns=[self.time_name, self.target_name, self.id_name]
        )

        for var in df_filtered:
            res_espect_time = {}
            for time in self.df[self.time_name].unique():
                df_time = self.df[self.df[self.time_name] == time]

                res_espec = {}

                if df_time[var].dtypes == "object" or len(df_time[var].unique()) <= 10:
                    df_time = self.df[self.df[self.time_name] == time]

                    res_espec = {
                        "number_of_missing": int(df_time[var].isnull().sum()),
                        "number_of_zeros": int((df_time[var] == 0).sum()),
                        "sum": None,
                        "mean": None,
                        "median": None,
                        "number_per_quintile": {
                            category: int((df_time[var] == category).sum())
                            for category in df_time[var].unique()
                        },
                    }

                elif df_time[var].dtype in ["int64", "float64"]:
                    quintil_cut_points[var] = (
                        self.df[var].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
                    )
                    res_espec = {
                        "number_of_missing": int(df_time[var].isnull().sum()),
                        "number_of_zeros": int((df_time[var] == 0).sum()),
                        "sum": float(df_time[var].sum()),
                        "mean": float(round(df_time[var].mean(), 2)),
                        "median": float(df_time[var].median()),
                        "number_per_quintile": {
                            f"<= {round(quintil_cut_points[var][0.2], 0)}": int(
                                (df_time[var] <= quintil_cut_points[var][0.2]).sum()
                            ),
                            f"<= {round(quintil_cut_points[var][0.4], 0)}": int(
                                (
                                    (df_time[var] > quintil_cut_points[var][0.2])
                                    & (df_time[var] <= quintil_cut_points[var][0.4])
                                ).sum()
                            ),
                            f"<= {round(quintil_cut_points[var][0.6], 0)}": int(
                                (
                                    (df_time[var] > quintil_cut_points[var][0.4])
                                    & (df_time[var] <= quintil_cut_points[var][0.6])
                                ).sum()
                            ),
                            f"<= {round(quintil_cut_points[var][0.8], 0)}": int(
                                (
                                    (df_time[var] > quintil_cut_points[var][0.6])
                                    & (df_time[var] <= quintil_cut_points[var][0.8])
                                ).sum()
                            ),
                            f"> {round(quintil_cut_points[var][0.8], 0)}": int(
                                (df_time[var] > quintil_cut_points[var][0.8]).sum()
                            ),
                        },
                    }
                res_espect_time[time] = res_espec
            res_espect_var[var] = res_espect_time
        return res_espect_var

    def variables_estability(self):
        psi_tot_results = {}

        ks_tot_results = {}

        unique_times = np.sort(self.df[self.time_name].unique())
        df_filtered = self.df.drop(
            columns=[self.time_name, self.target_name, self.id_name]
        )

        for var in df_filtered:
            psi_results = {}
            # Calculate psi for every month in the dataframe
            for i in range(len(unique_times) - 1):
                base_period = unique_times[i]
                comparison_period = unique_times[i + 1]
                psi_value = UtilsCalc.calculate_psi(
                    self.df, self.time_name, var, base_period, comparison_period
                )
                psi_results[f"{base_period} vs {comparison_period}"] = psi_value

            # Appending the results of the variable on the psi dictionary
            psi_tot_results[var] = psi_results

            if self.df[var].dtypes in ["int64", "float64"]:
                ks_results = {}
                # Calculate KS for every month in the dataframe
                for i in range(len(unique_times) - 1):
                    base_period = unique_times[i]
                    comparison_period = unique_times[i + 1]
                    ks_value = UtilsCalc.calculate_ks(
                        df=self.df,
                        time_column=self.time_name,
                        value_column=var,
                        base_period=base_period,
                        comparison_period=comparison_period,
                    )
                    ks_results[f"{base_period} vs {comparison_period}"] = ks_value

                # Appending the results of the variable on the psi dictionary
                ks_tot_results[var] = ks_results

        psi_ = sorted(
            psi_tot_results.items(), key=lambda x: max(x[1].values()), reverse=True
        )[: self.top_n]

        ks_ = sorted(
            ks_tot_results.items(), key=lambda x: max(x[1].values()), reverse=True
        )[: self.top_n]

        return psi_, ks_

    def variables_fillment(self):
        # Convert the number of missing values per variable to a dataframe
        df_missing = pd.DataFrame(
            list(self.df.isnull().sum().to_dict().items()),
            columns=["Variable", "Missing"],
        )

        # Sort the dataframe by the number of missing values in descending order
        df_missing_sorted = df_missing.sort_values(by="Missing", ascending=False)

        # Calculate the percentage of missing values for each variable
        df_missing_sorted["Missing (%)"] = (
            df_missing_sorted["Missing"] / self.df.shape[0]
        ) * 100


        df_missing_sorted["Missing"] = df_missing_sorted["Missing (%)"].apply(lambda x: f"{x:.0f}")
        df_missing_sorted["Missing (%)"] = df_missing_sorted["Missing (%)"].apply(lambda x: f"{x:.2f}%")

        miss_ = df_missing_sorted.head(self.top_n).to_dict()

        # Convert the number of Zero values per variable to a dataframe
        df_zero = pd.DataFrame(
            list((self.df == 0).sum().to_dict().items()),
            columns=["Variable", "Zero"],
        )

        # Sort the dataframe by the number of Zero values in descending order
        df_zero_sorted = df_zero.sort_values(by="Zero", ascending=False)

        # Calculate the percentage of missing values for each variable
        df_zero_sorted["Zero (%)"] = (
            df_zero_sorted["Zero"] / self.df.shape[0]
        ) * 100

        df_zero_sorted["Zero"] = df_zero_sorted["Zero (%)"].apply(lambda x: f"{x:.0f}")
        df_zero_sorted["Zero (%)"] = df_zero_sorted["Zero (%)"].apply(lambda x: f"{x:.2f}%")

        zero_ = df_zero_sorted.head(self.top_n).to_dict()

        return miss_, zero_

    def promising_features(self):
        iv_df = UtilsCalc.calculate_information_value(self.df, self.target_name)
        mi_df = UtilsCalc.calculate_mutual_information(self.df, self.target_name)

        # Select the top features based on Information Value
        top_iv_features = iv_df.to_dict()

        iv_ = sorted(
            top_iv_features["Information Value"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_n]

        # Select the top features based on Mutual Information

        top_mi_features = dict(zip(mi_df["Feature"], mi_df["Mutual Information"]))

        mi_ = sorted(top_mi_features.items(), key=lambda x: x[1], reverse=True)[:5]

        return iv_, mi_

    def report(self, report_version: str):
        ### OVERVIEW

        sandeda = SandEDA(
            self.df, self.target_name, self.time_name, self.id_name, self.top_n
        )
        res_general = sandeda.calc_general()

        overview_target_metric_time = res_general["target_general"][
            "target_metric_time"
        ]

        # Create the bar plot using Altair
        chart = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "Date": list(overview_target_metric_time.keys()),
                        "%": list(overview_target_metric_time.values()),
                    }
                )
            )
            .mark_bar()
            .encode(x=alt.X("Date", sort="ascending"), y=alt.Y("%"))
            .properties(width=100, height=100)
            .configure_axis(labelAngle=45)
            .configure_title(fontSize=10)
        )

        overview_tab_general = pd.DataFrame(
            list(res_general["dataset_general"].items()),
            columns=["Description", "Value"],
            index=None,
        )
        overview_tab_general["Value"] = overview_tab_general["Value"].apply(
            lambda x: f"{x:,.0f}")
        overview_tab_general = overview_tab_general.to_html(index=False, border=0)

        overview_tab_full = pd.DataFrame(res_general["missing_zero"]).to_html(
            index=False, border=0
        )

        overview_target_name = res_general["target_general"]["target_name"]

        overview_target_metric = round((
            res_general["target_general"]["number_of_one"]
            / (
                res_general["target_general"]["number_of_zero"]
                + res_general["target_general"]["number_of_one"]
            )
        ) * 100, 2)

        overview_tgt_graph_json = chart.to_json()

        ### VARIABLES

        iv_, mi_ = sandeda.promising_features()
        psi_, ks_ = sandeda.variables_estability()
        miss_, zero_ = sandeda.variables_fillment()

        var_tab_ks = pd.DataFrame(
            {
            "Variable": [var for var, _ in ks_],
            "KS": [round(max(value.values()), 3) for _, value in ks_],
            }
        ).to_html(index=False, border=0)

        var_tab_psi = pd.DataFrame(
            {
                "Variable": [var for var, _ in psi_],
                "PSI": [round(max(value.values()), 3) for _, value in psi_],
            }
        ).to_html(index=False, border=0)

        var_tab_iv = pd.DataFrame(
            {"Variable": [var for var, _ in iv_], "IV": [round(value, 3) for _, value in iv_]}
        ).to_html(index=False, border=0)

        var_tab_mi = pd.DataFrame(
            {"Variable": [var for var, _ in mi_], "MI": [round(value, 3) for _, value in mi_]}
        ).to_html(index=False, border=0)

        var_tab_miss = pd.DataFrame(miss_).to_html(index=False, border=0)
        var_tab_zero = pd.DataFrame(zero_).to_html(index=False, border=0)

        ### ESPECIFIC VARIABLES

        variables_espec = sandeda.variables_espec()

        variables_espec_time = sandeda.variables_espec_time()

        var_espec_content = {}

        vars_keys = variables_espec.keys() - {
            self.id_name,
            self.target_name,
            self.time_name,
        }

        for var_espec in vars_keys:
            hist_var = variables_espec[var_espec]["histogram"]

            decil_var = variables_espec[var_espec]["decil"]

            del (
                variables_espec[var_espec]["histogram"],
                variables_espec[var_espec]["decil"],
            )

            var_spec_tab_desc = pd.DataFrame(
                {
                    "Description": variables_espec[var_espec][
                        "descriptive_statistics"
                    ].keys(),
                    "Value": [
                        f"{value:,.0f}" if isinstance(value, (int, float)) else str(value) for value in variables_espec[var_espec][
                            "descriptive_statistics"
                        ].values()
                    ],
                }
            ).to_html(index=False, border=0)

            var_spec_tab_quant = pd.DataFrame(
                {
                    "Description": variables_espec[var_espec][
                        "quantile_statistics"
                    ].keys(),
                    "Value": [
                        f"{value:,.0f}" if isinstance(value, (int, float)) else str(value) for value in variables_espec[var_espec][
                            "quantile_statistics"
                        ].values()
                    ],
                }
            ).to_html(index=False, border=0)

            if variables_espec[var_espec]["descriptive_statistics"][
                "number_of_unique_values"
            ] <= 50 or variables_espec[var_espec]["descriptive_statistics"][
                "variable_type"
            ] not in ["int64", "float64"]:
                # Convert the histogram data to a dataframe
                hist_data = pd.DataFrame(
                    {
                        "Interval": list(hist_var.keys()),
                        "Count": list(hist_var.values()),
                    }
                )
            else:
                hist_data = pd.DataFrame(
                    {
                        "Interval": [
                            f"{interval.left:.1f}" for interval in hist_var.keys()
                        ],
                        "Count": list(hist_var.values()),
                    }
                )

            # Create the bar plot using Altair
            hist_chart = (
                alt.Chart(hist_data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Interval",
                        title="",
                        sort=None,
                        axis=alt.Axis(labels=True, labelOverlap=True, labelFontSize=9),
                    ),
                    y=alt.Y("Count", title="Qty", axis=alt.Axis(labels=False)),
                )
                .properties(width=350, height=200, title="Histogram")
                .configure_axis(labelAngle=45)
                .configure_title(fontSize=14)
                .interactive(False)
            )  # Disable interactive features

            var_spec_hist_graph_json = hist_chart.to_json()

            decil_data = pd.DataFrame(
                {"Decil": list(decil_var.index), "Target": list(np.round(decil_var.values, 2))}
            )

            # Create the bar plot using Altair
            decil_chart = (
                alt.Chart(decil_data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Decil",
                        title="Decil",
                        sort=None,
                        axis=alt.Axis(labels=True, labelOverlap=True, labelFontSize=9),
                    ),
                    y=alt.Y("Target", title="% Target", axis=alt.Axis(labels=False)),
                )
                .properties(width=350, height=200, title="Target mean per Decil")
                .configure_axis(labelAngle=45)
                .configure_title(fontSize=12)
                .interactive(False)
            )  # Disable interactive features

            var_spec_decil_graph_json = decil_chart.to_json()

            # Create a dataframe with the number_per_quintile values for each time period
            df_quintile_time = pd.DataFrame(
                {
                    "Date": list(variables_espec_time[var_espec].keys()),
                    "number_per_quintile": [
                        variables_espec_time[var_espec][date]["number_per_quintile"]
                        for date in variables_espec_time[var_espec].keys()
                    ],
                }
            )

            # Expand the number_per_quintile dictionary into separate columns
            df_quintile_time = df_quintile_time.join(
                pd.DataFrame(
                    df_quintile_time.pop("number_per_quintile").tolist(),
                    index=df_quintile_time.index,
                )
            )

            # Melt the dataframe to have a long format suitable for Altair
            df_melted = df_quintile_time.melt(
                id_vars="Date", var_name="Quintile", value_name="Count"
            )

            # Calculate the percentage for each quintile
            df_melted["Percentage"] = df_melted.groupby("Date")["Count"].transform(
                lambda x: x / x.sum() * 100
            )

            # Create the 100% stacked column chart using Altair
            stacked_chart = (
                alt.Chart(df_melted)
                .mark_bar()
                .encode(
                    x=alt.X("Date", title="Date"),
                    y=alt.Y("Percentage", title="Percentage", stack="normalize"),
                    color=alt.Color("Quintile", title="Legend"),
                )
                .properties(width=800, height=200, title="Distribution Over Time")
                .configure_axis(labelAngle=45)
                .configure_title(fontSize=14)
                .configure_legend(orient="top")
                .interactive(False)
            )  # Disable interactive features

            var_spec_stacked_graph_json = stacked_chart.to_json()

            var_espec_content[var_espec] = {
                "tab_desc": var_spec_tab_desc,
                "tab_quant": var_spec_tab_quant,
                "hist": var_spec_hist_graph_json,
                "decil": var_spec_decil_graph_json,
                "hist_time": var_spec_stacked_graph_json,
            }

        ### JINJA TEMPLATE

        # Read the template from the template sheet
        template_path = os.path.join(os.path.dirname(__file__), "./template/sandEda_template.html")
        with open(template_path, "r") as file:
            sanEda_template = file.read()

        # Create a Jinja2 template object

        template = Template(sanEda_template)

        # Render the template with the data
        rendered_html = template.render(
            title=report_version,
            overview_tab_general=overview_tab_general,
            overview_tab_full=overview_tab_full,
            overview_target_metric=overview_target_metric,
            overview_target_name=overview_target_name,
            overview_tgt_graph_json=overview_tgt_graph_json,
            var_tab_iv=var_tab_iv,
            var_tab_mi=var_tab_mi,
            var_tab_miss=var_tab_miss,
            var_tab_zero=var_tab_zero,
            var_tab_psi=var_tab_psi,
            var_tab_ks=var_tab_ks,
            specific_variables=var_espec_content,
        )

        # Save the rendered HTML to a file
        with open(f"SandEDA_{report_version}.html", "w") as file:
            file.write(rendered_html)