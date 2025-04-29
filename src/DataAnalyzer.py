
class DataAnalyzer:
    """Class for analyzing copula model data."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a dataset.
        
        Args:
            data: DataFrame containing the data
        """
        self.data = data

    def plot_scatter_matrix(self, figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """
        Create a scatter matrix plot to visualize relationships.
        
        Args:
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        sns.pairplot(self.data, diag_kind='kde')
        plt.tight_layout()
        return fig

    def plot_joint_distribution(self, x_var: str, y_var: str, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a joint distribution plot for two variables.
        
        Args:
            x_var: Name of the x variable
            y_var: Name of the y variable
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        sns.jointplot(data=self.data, x=x_var, y=y_var, kind='kde')
        plt.tight_layout()
        return fig

    def compute_correlations(self) -> pd.DataFrame:
        """
        Compute multiple correlation measures.
        
        Returns:
            DataFrame with different correlation measures
        """
        # Pearson correlation (linear)
        pearson_corr = self.data.corr(method='pearson')

        # Spearman correlation (monotonic)
        spearman_corr = self.data.corr(method='spearman')

        # Kendall's tau correlation (concordance)
        kendall_corr = self.data.corr(method='kendall')

        correlations = {
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'Kendall': kendall_corr
        }

        return correlations

    def compute_tail_dependence(self, x_var: str, y_var: str, threshold: float = 0.9) -> Dict[str, float]:
        """
        Estimate upper and lower tail dependence coefficients.
        
        Args:
            x_var: Name of the first variable
            y_var: Name of the second variable
            threshold: Quantile threshold for tail dependence
            
        Returns:
            Dictionary with upper and lower tail dependence coefficients
        """
        # Convert data to uniform scale (empirical CDF)
        x = self.data[x_var].values
        y = self.data[y_var].values

        n = len(x)
        rank_x = stats.rankdata(x) / (n + 1)
        rank_y = stats.rankdata(y) / (n + 1)

        # Upper tail dependence
        upper_mask = (rank_x > threshold) & (rank_y > threshold)
        upper_count = np.sum(upper_mask)
        upper_coef = upper_count / (n * (1 - threshold))

        # Lower tail dependence
        lower_mask = (rank_x < (1 - threshold)) & (rank_y < (1 - threshold))
        lower_count = np.sum(lower_mask)
        lower_coef = lower_count / (n * (1 - threshold))

        return {
            'upper_tail_dependence': upper_coef,
            'lower_tail_dependence': lower_coef
        }
