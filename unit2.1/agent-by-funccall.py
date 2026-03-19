from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, Tool, tool, VisitWebpageTool
from typing import List, Dict, Union, Tuple
import math

 # 计算参数离散函数值
@tool
def calculate_discrete_function(numbers: List[Union[int, float]], operation: str) -> Dict[str, Union[float, int, List[float]]]:
    """
    Calculates discrete function values for a list of numbers.

    Args:
        numbers: A list of integers or floating-point numbers.
        operation: The type of operation to perform. Supported values are:
            - "sum": Sum of all numbers
            - "mean": Arithmetic mean
            - "median": Median value
            - "variance": Population variance
            - "std": Population standard deviation
            - "minmax": Minimum and maximum values
            - "freq": Frequency distribution
            - "normalize": Normalized values (between 0 and 1)
            - "cumsum": Cumulative sum
            - "sort": Sorted list

    Returns:
        A dictionary containing the result. The structure varies by operation.
        Common keys include:
            - "result": The main result (for simple operations)
            - "values": List of values (for operations returning lists)
            - "min" and "max": For minmax operation
            - "frequencies": For freq operation

    Raises:
        ValueError: If the input list is empty or operation is not supported.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty.")
    
    if operation == "sum":
        return {"result": sum(numbers)}
    
    elif operation == "mean":
        return {"result": sum(numbers) / len(numbers)}
    
    elif operation == "median":
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        if n % 2 == 1:
            median_val = sorted_nums[n // 2]
        else:
            median_val = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        return {"result": median_val}
    
    elif operation == "variance":
        mean_val = sum(numbers) / len(numbers)
        variance_val = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
        return {"result": variance_val}
    
    elif operation == "std":
        mean_val = sum(numbers) / len(numbers)
        variance_val = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
        return {"result": math.sqrt(variance_val)}
    
    elif operation == "minmax":
        return {"min": min(numbers), "max": max(numbers)}
    
    elif operation == "freq":
        freq_dict = {}
        for num in numbers:
            key = num
            freq_dict[key] = freq_dict.get(key, 0) + 1
        return {"frequencies": freq_dict}
    
    elif operation == "normalize":
        min_val = min(numbers)
        max_val = max(numbers)
        if min_val == max_val:
            normalized = [0.0] * len(numbers)
        else:
            normalized = [(x - min_val) / (max_val - min_val) for x in numbers]
        return {"values": normalized}
    
    elif operation == "cumsum":
        cum_sum = []
        current_sum = 0
        for num in numbers:
            current_sum += num
            cum_sum.append(current_sum)
        return {"values": cum_sum}
    
    elif operation == "sort":
        return {"values": sorted(numbers)}
    
    else:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations are: sum, mean, median, variance, std, minmax, freq, normalize, cumsum, sort.")


# 计算二维数组的线性分布统计
@tool
def calculate_linear_distribution(x_values: List[Union[int, float]], 
                                 y_values: List[Union[int, float]] = None, 
                                 operation: str = "all") -> Dict[str, Union[float, Dict, Tuple]]:
    """
    Calculates linear distribution statistics for a set of numbers.

    Args:
        x_values: A list of integers or floating-point numbers (independent variable).
        y_values: Optional list for bivariate analysis. If None, uses index as x and x_values as y.
        operation: The type of operation to perform. Supported values are:
            - "all": Returns all linear distribution statistics
            - "correlation": Pearson correlation coefficient
            - "regression": Linear regression coefficients (slope, intercept)
            - "trend": Trend direction (increasing/decreasing) and strength
            - "r_squared": Coefficient of determination
            - "residuals": Residuals (errors) for each data point
            - "normality": Tests for normality of residuals

    Returns:
        A dictionary containing the requested statistics.

    Raises:
        ValueError: If the input lists are invalid or operation is not supported.
    """
    # Validate input
    if not x_values:
        raise ValueError("x_values list cannot be empty.")
    
    # Handle univariate case (use index as x)
    if y_values is None:
        y_values = x_values.copy()
        x_values = list(range(1, len(y_values) + 1))
    elif len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    elif len(x_values) < 2:
        raise ValueError("At least 2 data points are required.")
    
    n = len(x_values)
    
    # Helper function to calculate basic statistics
    def calculate_basic_stats(x_vals, y_vals):
        """Calculate basic statistics needed for linear analysis."""
        x_mean = statistics.mean(x_vals)
        y_mean = statistics.mean(y_vals)
        
        # Calculate sums
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)
        sum_y2 = sum(y * y for y in y_vals)
        
        return {
            'n': n,
            'x_mean': x_mean,
            'y_mean': y_mean,
            'sum_x': sum_x,
            'sum_y': sum_y,
            'sum_xy': sum_xy,
            'sum_x2': sum_x2,
            'sum_y2': sum_y2
        }
    
    # Helper function to calculate correlation
    def calculate_correlation(stats):
        """Calculate Pearson correlation coefficient."""
        numerator = stats['sum_xy'] - stats['n'] * stats['x_mean'] * stats['y_mean']
        denominator_x = stats['sum_x2'] - stats['n'] * stats['x_mean'] ** 2
        denominator_y = stats['sum_y2'] - stats['n'] * stats['y_mean'] ** 2
        
        if denominator_x <= 0 or denominator_y <= 0:
            return 0.0
        
        return numerator / math.sqrt(denominator_x * denominator_y)
    
    # Helper function to calculate linear regression
    def calculate_regression(stats):
        """Calculate linear regression coefficients (slope, intercept)."""
        numerator = stats['sum_xy'] - stats['n'] * stats['x_mean'] * stats['y_mean']
        denominator = stats['sum_x2'] - stats['n'] * stats['x_mean'] ** 2
        
        if denominator == 0:
            return None, None  # Vertical line
        
        slope = numerator / denominator
        intercept = stats['y_mean'] - slope * stats['x_mean']
        
        return slope, intercept
    
    # Calculate basic statistics
    stats = calculate_basic_stats(x_values, y_values)
    
    # Calculate correlation
    correlation = calculate_correlation(stats)
    
    # Calculate regression coefficients
    slope, intercept = calculate_regression(stats)
    
    # Prepare result dictionary
    result = {
        'n': stats['n'],
        'x_mean': stats['x_mean'],
        'y_mean': stats['y_mean'],
        'correlation': correlation
    }
    
    # Add regression results if available
    if slope is not None and intercept is not None:
        result['slope'] = slope
        result['intercept'] = intercept
        result['regression_equation'] = f"y = {slope:.4f}x + {intercept:.4f}"
        
        # Calculate R-squared
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
        ss_tot = sum((y - stats['y_mean']) ** 2 for y in y_values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        result['r_squared'] = r_squared
        
        # Calculate residuals
        residuals = [y - (slope * x + intercept) for x, y in zip(x_values, y_values)]
        result['residuals'] = residuals
        result['residual_mean'] = statistics.mean(residuals)
        result['residual_std'] = statistics.stdev(residuals) if len(residuals) > 1 else 0
        
        # Calculate predicted values
        predicted = [slope * x + intercept for x in x_values]
        result['predicted'] = predicted
        
        # Determine trend direction
        if slope > 0.1:
            trend = "strongly increasing"
        elif slope > 0.01:
            trend = "moderately increasing"
        elif slope > 0:
            trend = "slightly increasing"
        elif slope < -0.1:
            trend = "strongly decreasing"
        elif slope < -0.01:
            trend = "moderately decreasing"
        elif slope < 0:
            trend = "slightly decreasing"
        else:
            trend = "no trend (constant)"
        
        result['trend'] = trend
        result['trend_strength'] = abs(correlation)
    
    # Return based on requested operation
    if operation == "correlation":
        return {'correlation': correlation}
    
    elif operation == "regression":
        if slope is None or intercept is None:
            return {'error': 'Cannot calculate regression (vertical line or invalid data)'}
        return {
            'slope': slope,
            'intercept': intercept,
            'regression_equation': f"y = {slope:.4f}x + {intercept:.4f}"
        }
    
    elif operation == "trend":
        if slope is None:
            return {'trend': 'indeterminate'}
        return {
            'trend': result.get('trend', 'indeterminate'),
            'trend_strength': result.get('trend_strength', 0)
        }
    
    elif operation == "r_squared":
        return {'r_squared': result.get('r_squared', 0)}
    
    elif operation == "residuals":
        if 'residuals' not in result:
            return {'error': 'Regression coefficients not available'}
        return {
            'residuals': result['residuals'],
            'residual_mean': result['residual_mean'],
            'residual_std': result['residual_std']
        }
    
    elif operation == "normality":
        if 'residuals' not in result:
            return {'error': 'Residuals not available'}
        
        residuals = result['residuals']
        if len(residuals) < 3:
            return {'error': 'Insufficient data for normality test'}
        
        # Simple normality check (skewness and kurtosis)
        mean_res = statistics.mean(residuals)
        std_res = statistics.stdev(residuals) if len(residuals) > 1 else 0
        
        if std_res == 0:
            return {'normality': 'perfect_normal', 'skewness': 0, 'kurtosis': 0}
        
        # Calculate skewness
        skewness = sum(((r - mean_res) / std_res) ** 3 for r in residuals) / len(residuals)
        
        # Calculate kurtosis (excess kurtosis)
        kurtosis = sum(((r - mean_res) / std_res) ** 4 for r in residuals) / len(residuals) - 3
        
        # Determine normality
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            normality = "approximately_normal"
        elif abs(skewness) < 1 and abs(kurtosis) < 2:
            normality = "moderately_normal"
        else:
            normality = "not_normal"
        
        return {
            'normality': normality,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    elif operation == "all":
        return result
    
    else:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations are: all, correlation, regression, trend, r_squared, residuals, normality.")


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""
    
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }
    
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        
        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")


# Alfred, the butler, preparing the menu for the party
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(), 
        VisitWebpageTool(),
        calculate_discrete_function,
        calculate_linear_distribution,
        SuperheroPartyThemeTool(),
	    FinalAnswerTool()
    ], 
    model=InferenceClientModel(),
    max_steps=10,
    verbosity_level=2
)

agent.run("Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")