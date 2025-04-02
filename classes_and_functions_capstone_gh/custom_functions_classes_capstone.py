from typing import Optional, List, Literal
import re
from collections import Counter
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)


#====================================================================#
#                               Classes                              #
#====================================================================#


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, minimal_value = 0):
        """
        Parameters:
        - columns: list of str
          List of column names to apply the log(1 + x) transformation.
        """
        self.columns = columns
        self.minimal_value = minimal_value

    def fit(self, X, y=None):
        # No fitting necessary for this transformer, so we just return self
        return self

    def transform(self, X):
        """
        Apply the log(1 + x) transformation to the specified columns.
        
        Parameters:
        - X: pd.DataFrame
          The input DataFrame.
        
        Returns:
        - X_transformed: pd.DataFrame
          The transformed DataFrame.
        """
        # Make a copy of the DataFrame to avoid modifying the original one
        X_transformed = X.copy()
        
        # Apply log(1 + x) to specified columns
        if self.columns is not None:
            for col in self.columns:
                # Ensure column exists in the DataFrame
                if col in X_transformed.columns:
                    X_transformed[col] = np.log1p(np.add(X_transformed[col], np.abs(self.minimal_value)))

        return X_transformed

class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed


class CustomOutlierRemoverNormal(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.numeric_cols = None
        self._outliers = None

    # This function identifies the numerical columns
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns
        return self

    def transform(self, X):
        if self.numeric_cols is None:
            raise ValueError("Call 'fit' before 'transform'.")

        # Make a copy of numerical columns
        X_transformed = X.copy()

        z_scores = stats.zscore(X_transformed[self.numeric_cols])

        # Concat with non-numerical columns
        self._outliers = (abs(z_scores) > self.threshold).any(axis=1)
        return X_transformed[~self._outliers]

    @property
    def outliers(self):
        return self._outliers
    

class CustomOutlierRemoverInterquartile(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.numeric_cols = None
        self._outliers = None

    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns
        return self

    def transform(self, X):
        if self.numeric_cols is None:
            raise ValueError("Call 'fit' before 'transform'.")

        # Make a copy of numerical columns
        X_transformed = X.copy()

        self._outliers = pd.Series([False] * len(X), index=X.index)
        for col in self.numeric_cols:
            Q1 = X_transformed[col].quantile(0.25)
            Q3 = X_transformed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            col_outliers = (X_transformed[col] < lower_bound) | (X_transformed[col] > upper_bound)
            self._outliers = self._outliers | col_outliers

        return X_transformed[~self._outliers]

    @property
    def outliers(self):
        return self._outliers


#====================================================================#
#                              Functions                             #
#====================================================================#


def show_pca_weights(pca_df: pd.DataFrame, pca_3d: PCA, df_preprocessed: pd.DataFrame):

    """
    Generate a DataFrame showing the PCA weights (loadings) for each component.

    Parameters
    ----------
    pca_df : pd.DataFrame
        DataFrame containing the PCA-transformed data with samples as rows and principal components as columns.
    
    pca_3d : PCA
        Fitted PCA object from scikit-learn containing the principal components and explained variance.
    
    df_preprocessed : pd.DataFrame
        Original preprocessed DataFrame that was used as input for the PCA, with features as columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loadings (weights) of each feature for each principal component. The columns
        represent the principal components (PC1, PC2, etc.), and the rows represent the original features.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3],
    >>>     'feature2': [4, 5, 6],
    >>>     'feature3': [7, 8, 9]
    >>> })
    >>> pca = PCA(n_components=2)
    >>> pca_transformed = pca.fit_transform(df)
    >>> pca_df = pd.DataFrame(pca_transformed)
    >>> loadings = show_pca_weights(pca_df, pca, df)
    >>> print(loadings)
               PC1       PC2
    feature1 -0.57735 -0.70711
    feature2 -0.57735  0.00000
    feature3 -0.57735  0.70711
    """

    component_names = [f"PC{i+1}" for i in range(pca_df.shape[1])]
    pca_df = pd.DataFrame(pca_df, columns=component_names)

    loadings = pd.DataFrame(
        pca_3d.components_.T,
        columns=component_names,
        index=df_preprocessed.columns,
    )
    return loadings


def plot_explained_variance(pca_3d: PCA):

    """
    Plot the explained variance ratio for each principal component in a PCA model.

    Parameters
    ----------
    pca_3d : PCA
        Fitted PCA object from scikit-learn containing the principal components and explained variance ratios.

    Returns
    -------
    None
        This function does not return anything. It prints the explained variance ratio for each principal component
        and displays a bar plot of the explained variance ratio.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3],
    >>>     'feature2': [4, 5, 6],
    >>>     'feature3': [7, 8, 9]
    >>> })
    >>> pca = PCA(n_components=3)
    >>> pca.fit(df)
    >>> plot_explained_variance(pca)
    Explained variance ratio for each component (3D PCA):
    Principal Component 1: 1.0000
    Principal Component 2: 0.0000
    Principal Component 3: 0.0000
    Sum of variances: 1.0
    """

    explained_variance_3d = pca_3d.explained_variance_ratio_
    print("Explained variance ratio for each component (3D PCA):")
    var_sum = 0
    for i, var in enumerate(explained_variance_3d, 1):
        print(f"Principal Component {i}: {var:.4f}")
        var_sum+=var
    print('Sum of variances:', round(var_sum, 3))

    plt.bar(range(1, len(explained_variance_3d)+1), explained_variance_3d, alpha=0.7, color='b')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components (3D PCA)')
    plt.xticks(np.arange(1, len(explained_variance_3d)+1))

    plt.show()


def plot_algo3d_interactive(df: np.ndarray, target: pd.Series, algo_name: str, target_name: str, dot_size=3):
    """
    Plot an interactive 3D scatter plot using plotly with adjustable dot size.

    Parameters
    ----------
    df : np.ndarray
        Array containing the transformed data with samples as rows and the first three components as columns.
    
    target : pd.Series
        Series containing the target labels corresponding to the samples in the transformed data.
    
    algo_name : str
        Name of the algorithm used to transform the data (e.g., 't-SNE').
    
    target_name : str
        Name of the target variable (e.g., 'Clusters').

    dot_size : int or float, optional
        The size of the dots in the plot. Default is 5.

    Returns
    -------
    None
        This function does not return anything. It displays an interactive 3D scatter plot.
    
    Examples
    --------
    >>> plot_algo3d_interactive(tsne_data, target_series, 't-SNE', 'Clusters', dot_size=10)
    """
    # Create a DataFrame for plotly
    plot_df = pd.DataFrame({
        f'{algo_name} Component 1': df[:, 0],
        f'{algo_name} Component 2': df[:, 1],
        f'{algo_name} Component 3': df[:, 2],
        target_name: target
    })

    # Create a 3D scatter plot
    fig = px.scatter_3d(
        plot_df,
        x=f'{algo_name} Component 1',
        y=f'{algo_name} Component 2',
        z=f'{algo_name} Component 3',
        color=target_name,
        title=f'Interactive 3D {algo_name} Plot',
        color_continuous_scale='Viridis',
        size_max=dot_size  # This sets the maximum size of the dots
    )

    # Customize layout for better interaction
    fig.update_traces(marker=dict(size=dot_size))  # Set a uniform dot size for all points
    fig.update_layout(
        scene=dict(
            xaxis_title=f'{algo_name} Component 1',
            yaxis_title=f'{algo_name} Component 2',
            zaxis_title=f'{algo_name} Component 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the plot
    fig.show()


def plot_algo3d(df: np.ndarray, target: pd.Series, algo_name: str, target_name: str):
    """
    Plot a 3D scatter plot with different viewing angles.

    Parameters
    ----------
    df : np.ndarray
        DataFrame containing the PCA-transformed data with samples as rows and the first three principal components as columns.
    
    target : pd.Series
        Series containing the target labels corresponding to the samples in the PCA-transformed data.
    
    algo_name : str
        Name of the algorithm used to transform the data (e.g., 'PCA').
    
    target_name : str
        Name of the target variable (e.g., 'class').

    Returns
    -------
    None
        This function does not return anything. It displays 3D scatter plots of the PCA components colored by the target labels.
    
    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4],
    >>>     'feature2': [4, 5, 6, 7],
    >>>     'feature3': [7, 8, 9, 10],
    >>>     'target': [0, 1, 0, 1]
    >>> })
    >>> X = df.drop('target', axis=1)
    >>> y = df['target']
    >>> pca = PCA(n_components=3)
    >>> pca_transformed = pca.fit_transform(X)
    >>> plot_algo3d(pca_transformed, y, 'PCA', 'class')
    """

    labels = target

    for perm in [[90, 90], [0, 90], [0, 0]]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df[:, 0], df[:, 1], df[:, 2], c=labels, cmap='viridis', marker='o')
        ax.set_title(f'3D {algo_name} with {target_name} angles=({perm[0]},{perm[1]})')
        ax.set_xlabel(f'{algo_name} Component 1')
        ax.set_ylabel(f'{algo_name} Component 2')
        ax.set_zlabel(f'{algo_name} Component 3')
        ax.view_init(elev=perm[0], azim=perm[1])
        legend1 = ax.legend(*scatter.legend_elements(), title=target_name)
        ax.add_artist(legend1)
        plt.show()



def plot_class_distribution(target: pd.Series, target_name: str) -> None:

    """
    Plot the distribution of class labels in a target Series.

    Parameters
    ----------
    target : pd.Series
        Series containing the target labels.
    
    target_name : str
        Name of the target variable, used for labeling the x-axis of the plot.

    Returns
    -------
    None
        This function does not return anything. It prints the distribution of class labels and displays a bar plot.
    
    Examples
    --------
    >>> import pandas as pd
    >>> target = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    >>> plot_class_distribution(target, 'Class')
    Class=0, n=5 (50.000%)
    Class=1, n=5 (50.000%)
    """

    count = Counter(target)
    for label, num_samples in count.items():
        percentage = num_samples / len(target) * 100
        print('Class=%d, n=%d (%.3f%%)' % (label, num_samples, percentage))

    # Plot the counts as a bar graph
    warnings.filterwarnings("ignore")
    sns.barplot(x=list(count.keys()), y=list(count.values()))

    # Add labels and title
    plt.xlabel(target_name)
    plt.ylabel('Count')
    plt.title('Class Label Distribution')

    # Show the plot
    plt.show()


def plot_pairplots_kde_hue(df_all: pd.DataFrame, binary_columns: list):

    """
    Generate pair plots with KDE on the diagonal and hue based on binary columns.

    Parameters
    ----------
    df_all : pd.DataFrame
        DataFrame containing all features, including binary columns for hue distinction.
    
    binary_columns : list
        List of column names in `df_all` that are binary and used for hue distinction in the pair plots.

    Returns
    -------
    None
        This function does not return anything. It displays pair plots with KDE on the diagonal for each binary column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4],
    >>>     'feature2': [4, 5, 6, 7],
    >>>     'binary1': [0, 1, 0, 1],
    >>>     'binary2': [1, 0, 1, 0]
    >>> })
    >>> plot_pairplots_kde_hue(df, ['binary1', 'binary2'])
    """

    df_nonbinary = df_all.drop(binary_columns, axis=1)
    for column in binary_columns:
        raw_data_pairplot = pd.concat([df_nonbinary, df_all[column]], axis=1)
        sns.pairplot(raw_data_pairplot, diag_kind='kde', hue=column, height=2, aspect=1).figure.suptitle(f'Pairplot with distinction for {column}', y=1.05)
        plt.show()
        raw_data_pairplot = raw_data_pairplot.drop(column, axis=1)


def plot_histograms_nonbinary_logarithmic(df: pd.DataFrame, columns_to_use: list, figure_size: tuple):
    
    """
    Plot logarithmic histograms for non-binary columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be plotted.
    
    columns_to_use : list
        List of column names to plot histograms for.
    
    figure_size : tuple
        Size of the figure for the histograms.

    Returns
    -------
    None
        This function does not return anything. It displays histograms with logarithmic scaling for the specified columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 10, 100, 1000],
    >>>     'feature2': [2, 20, 200, 2000],
    >>>     'binary1': [0, 1, 0, 1]
    >>> })
    >>> plot_histograms_nonbinary_logarithmic(df, ['feature1', 'feature2'], (12, 6))
    """

    plt.figure(figsize=figure_size)
    for i, column in enumerate(columns_to_use):
        plt.subplot(1, 2, i+1)
        sns.histplot(np.log1p(df[column]), bins=len(df[column])//15, color='blue', edgecolor='black', kde=True)
        plt.title(f'Histogram | logarithmic {column}')
    plt.tight_layout()
    plt.show()


def plot_histograms_nonbinary(df: pd.DataFrame, figure_size: tuple):

    """
    Plot histograms for each non-binary column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be plotted.
    
    figure_size : tuple
        Size of the figure for the histograms.

    Returns
    -------
    None
        This function does not return anything. It displays histograms for each column in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4],
    >>>     'feature2': [5, 6, 7, 8],
    >>>     'feature3': [9, 10, 11, 12]
    >>> })
    >>> plot_histograms_nonbinary(df, (12, 8))
    """

    plt.figure(figsize=figure_size)
    for i, column in enumerate(df.columns):
        plt.subplot(2, 4, i+1)
        sns.histplot(df[column], bins=len(df[column])//15, color='blue', edgecolor='black', kde=True)
        plt.title(f'Histogram | {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def plot_violin_with_binary_hue(df_all: pd.DataFrame, binary_columns: list, figure_size: tuple):
    
    """
    Plot violin plots for each non-binary column in a DataFrame with binary columns as hue.

    Parameters
    ----------
    df_all : pd.DataFrame
        DataFrame containing all features, including binary columns for hue distinction.
    
    binary_columns : list
        List of column names in `df_all` that are binary and used for hue distinction in the violin plots.
    
    figure_size : tuple
        Size of the figure for the violin plots.

    Returns
    -------
    None
        This function does not return anything. It displays violin plots for each non-binary column with binary columns as hue.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4],
    >>>     'feature2': [5, 6, 7, 8],
    >>>     'binary1': [0, 1, 0, 1],
    >>>     'binary2': [1, 0, 1, 0]
    >>> })
    >>> plot_violin_with_binary_hue(df, ['binary1', 'binary2'], (12, 6))
    """

    for column in df_all.drop(binary_columns, axis=1).columns:
        plt.figure(figsize=figure_size)
        for i, bi_column in enumerate(binary_columns):
            plt.subplot(1, len(binary_columns), i+1)
            sns.violinplot(x=df_all[bi_column], y=df_all[column])
            plt.title(rf'{column} | {bi_column}')
            plt.xlabel(bi_column)
            plt.ylabel(column)
            plt.xticks(fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_indices_relation(df: pd.DataFrame, figure_size: tuple):

    """
    Plot scatter plots of each feature in a DataFrame against the DataFrame's index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be plotted.
    
    figure_size : tuple
        Size of the figure for the scatter plots.

    Returns
    -------
    None
        This function does not return anything. It displays scatter plots of each feature against the index.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4, 5],
    >>>     'feature2': [5, 6, 7, 8, 9],
    >>>     'feature3': [9, 10, 11, 12, 13]
    >>> })
    >>> plot_indices_relation(df, (12, 10))
    """

    plt.figure(figsize=figure_size)
    for i, feature in enumerate(df.columns):
        plt.subplot(5, 3, i+1)
        plt.scatter(df.index, df[feature], c='blue', s=3)
        plt.xlabel("index")
        plt.ylabel(feature)
        plt.title(f"{feature} and index")
    plt.tight_layout()
    plt.show()


def plot_violin_features(df: pd.DataFrame, figure_size: tuple):

    """
    Plot violin plots for each feature in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be plotted.
    
    figure_size : tuple
        Size of the figure for the violin plots.

    Returns
    -------
    None
        This function does not return anything. It displays violin plots for each feature in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4, 5],
    >>>     'feature2': [5, 6, 7, 8, 9],
    >>>     'feature3': [9, 10, 11, 12, 13]
    >>> })
    >>> plot_violin_features(df, (12, 8))
    """

    plt.figure(figsize=figure_size)
    for i, column in enumerate(df.columns):
        plt.subplot(2, 4, i+1)
        sns.violinplot(data=df[column])
        plt.title(f'Distribution of feature - {column}')
        plt.xlabel('Feature')
        plt.ylabel('Value')
        plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def make_mi_scores(features, target):

    """
    Compute and sort Mutual Information (MI) scores between each feature and the target variable.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing the feature columns for which MI scores are to be computed.
    
    target : pd.Series
        Series containing the target variable with which MI scores are calculated.

    Returns
    -------
    pd.Series
        Series containing MI scores for each feature, sorted in descending order. The index represents the feature names, and the values represent the MI scores.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.feature_selection import mutual_info_regression
    >>> X = pd.DataFrame({
    >>>     'feature1': [1, 2, 3, 4, 5],
    >>>     'feature2': [5, 6, 7, 8, 9]
    >>> })
    >>> y = pd.Series([0, 1, 0, 1, 0])
    >>> mi_scores = make_mi_scores(X, y)
    >>> print(mi_scores)
    feature1    0.345
    feature2    0.123
    Name: MI Scores, dtype: float64
    """

    mi_scores = mutual_info_regression(features, target, random_state=3721)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):

    """
    Plot a horizontal bar chart of Mutual Information (MI) scores.

    Parameters
    ----------
    scores : pd.Series
        Series containing Mutual Information scores, with feature names as the index and MI scores as the values.

    Returns
    -------
    None
        This function does not return anything. It displays a horizontal bar chart of the MI scores.

    Examples
    --------
    >>> import pandas as pd
    >>> mi_scores = pd.Series({
    >>>     'feature1': 0.345,
    >>>     'feature2': 0.123
    >>> })
    >>> plot_mi_scores(mi_scores)
    """

    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.figure(dpi=100, figsize=(8, 5))
    plt.show()


def unique_column_content_check(df: pd.DataFrame):

    """
    Returns the unique values and their count for each categorical column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for unique values in its categorical columns.

    Returns
    -------
    dict
        A dictionary where keys are the column names and values are tuples containing the unique values and their count.

    Description
    -----------
    This function iterates over each column in the provided DataFrame. For columns with a data type of `object` or `str`, 
    it identifies and stores the unique values present in that column along with the count of these unique values. The 
    results are returned in the form of a dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': ['apple', 'banana', 'apple'],
    ...     'B': [1, 2, 3],
    ...     'C': ['dog', 'cat', 'dog']
    ... })
    >>> unique_column_content_check(data)
    {'A': (array(['apple', 'banana'], dtype=object), 2), 
     'C': (array(['dog', 'cat'], dtype=object), 2)}
    """

    store_unique = {}
    for column in df.columns:
        unique_values = np.sort(df[column].unique())
        store_unique[column] = (unique_values, len(unique_values))
        

    return store_unique


def corr_matrix_dataframe(df, method='pearson'):
  
  """
    Computes the absolute correlation matrix for a DataFrame and returns it as a DataFrame sorted by correlation strength.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing numerical data.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the absolute correlation matrix sorted by correlation strength.

    Description
    -----------
    This function calculates the absolute correlation matrix for the input DataFrame, excluding the diagonal.
    It then sorts the correlations in descending order and returns them as a DataFrame with a single column 'correlation'.
    The resulting DataFrame contains pairs of features and their absolute correlation values, sorted by correlation strength.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> corr_matrix = corr_matrix_dataframe(data)
    >>> print(corr_matrix)
          correlation
    A  B      1.000000
    B  C      1.000000
    A  C      1.000000
    """

  correlations = df.corr(method=method)
  np.fill_diagonal(correlations.values, np.nan)
  mask = np.tril(np.ones_like(correlations, dtype=bool))
  correlations[mask] = np.nan

  return pd.DataFrame(correlations.abs().unstack().dropna().sort_values(ascending=False), columns=['correlation'])


def skewness(df):

    """
    Calculate and print the skewness of each feature in a DataFrame.

    The skewness value indicates the asymmetry of the distribution of the data. Positive skewness means the distribution is right-skewed, 
    negative skewness means it is left-skewed, and zero skewness indicates a normal distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features for which skewness is to be computed.

    Returns
    -------
    None
        This function does not return anything. It prints the skewness and distribution type of each feature.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 2, 3, 4, 5, 6],
    >>>     'feature2': [7, 8, 8, 8, 9, 10, 10],
    >>>     'feature3': [1, 1, 1, 1, 1, 1, 1]
    >>> })
    >>> skewness(df)
    Feature: feature1 
    skewness = 0.725, right-skewed.

    Feature: feature2 
    skewness = 0.198, right-skewed.

    Feature: feature3 
    skewness = 0.000, normal distribution.
    """

    for col in df.columns:
        skewness = df[col].skew()

        if skewness > 0:
            print(f'Feature: {col} \n skewness = {skewness:.3f}, right-skewed.')
        elif skewness < 0:
            print(f'Feature: {col} \n skewness = {skewness:.3f}, left-skewed.')
        else:
            print(f'Feature: {col} \n skewness = {skewness:.3f}, normal distribution. \n\n')


def lda_transform_plot(X: pd.DataFrame, y: pd.Series, title: str) -> None:
    lda = LinearDiscriminantAnalysis()
    lda_transformed = lda.fit_transform(X, y)

    num_components = lda_transformed.shape[1]

    # Create a DataFrame with the LDA results
    lda_df = pd.DataFrame(lda_transformed, columns=[f'LD{i+1}' for i in range(num_components)])
    lda_df['target'] = y

    # Plot the results
    plt.figure(figsize=(10, 6))

    # num of component verification
    if num_components == 1:
        sns.scatterplot(x='LD1', y=[0] * len(lda_df), hue='target', data=lda_df, palette='viridis', s=100, alpha=0.7, edgecolor='k')
        plt.xlabel('LD1')
        plt.ylabel('')
    else:
        sns.scatterplot(x='LD1', y='LD2', hue='target', data=lda_df, palette='viridis', s=100, alpha=0.7, edgecolor='k')
        plt.xlabel('LD 1')
        plt.ylabel('LD 2')

    plt.title(title)
    plt.legend(title='Target')
    plt.show()