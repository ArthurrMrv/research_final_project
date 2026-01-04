import requests
import os
import dotenv
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from collections import defaultdict


class RTEAPI:
    BASE_URL = "https://digital.iservices.rte-france.com"
    TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
    API_BASE_PATH = "/open_api/generation_installed_capacities/v1"
    
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = self.get_access_token()

    def get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token using client credentials flow.
        """
        # RTE requires Basic Auth (Base64 encoded ID:Secret in header)
        # We also create a session to handle the Content-Type automatically
        
        # 1. Use the 'auth' parameter to handle Basic Auth headers automatically
        response = requests.post(
            self.TOKEN_URL,
            auth=(self.client_id, self.client_secret), # Encodes to Authorization: Basic ...
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            # Some RTE endpoints technically don't even need the data body, 
            # but strictly speaking OAuth2 requires grant_type.
            data={"grant_type": "client_credentials"} 
        )
        
        # 2. Add better error printing to see WHY it failed if it fails again
        if response.status_code != 200:
            print(f"Failed to get token: {response.status_code}")
            print(f"Response body: {response.text}")
            
        response.raise_for_status()
        token_info = response.json()
        return token_info["access_token"]
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an authenticated request to the RTE API.
        
        Args:
            endpoint: API endpoint path (e.g., '/capacities_per_production_type')
            params: Optional query parameters
            
        Returns:
            JSON response data
        """
        url = f"{self.BASE_URL}{self.API_BASE_PATH}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_capacities_per_production_type(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get installed capacities of production units of more than 1 MW.
        
        Args:
            start_date: Optional start date in ISO 8601 format with timezone (e.g., '2023-02-01T00:00:00+01:00')
            end_date: Optional end date in ISO 8601 format with timezone (e.g., '2023-02-14T00:00:00+01:00')
            
        Returns:
            JSON response with capacities per production type
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self._make_request("/capacities_per_production_type", params=params if params else None)
    
    def get_capacities_per_production_unit(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get installed capacities of power stations of more than 100 MW, located on French mainland (excluding Corsica).
        
        Args:
            start_date: Optional start date in ISO 8601 format with timezone (e.g., '2023-02-01T00:00:00+01:00')
            end_date: Optional end date in ISO 8601 format with timezone (e.g., '2023-02-14T00:00:00+01:00')
            
        Returns:
            JSON response with capacities per production unit
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self._make_request("/capacities_per_production_unit", params=params if params else None)
    
    def get_capacities_cpc(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get installed capacities of production units bound by purchase obligation agreements with EDF.
        
        Args:
            start_date: Optional start date in ISO 8601 format with timezone (e.g., '2023-02-01T00:00:00+01:00')
            end_date: Optional end date in ISO 8601 format with timezone (e.g., '2023-02-14T00:00:00+01:00')
            
        Returns:
            JSON response with CPC capacities
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self._make_request("/capacities_cpc", params=params if params else None)


def plot_capacities_per_production_type(data: Dict[str, Any], save_path: Optional[str] = None, show: bool = True):
    """
    Plot installed capacities per production type.
    
    Args:
        data: API response from get_capacities_per_production_type
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    if "capacities_per_production_type" not in data:
        print("Error: Invalid data structure")
        return
    
    capacities = data["capacities_per_production_type"]
    
    # Aggregate values by production type (sum across all departments/network connections)
    type_totals = defaultdict(float)
    
    for entry in capacities:
        prod_type = entry.get("production_type")
        value = entry.get("value")
        
        # Only process entries with both production_type and a valid value
        if prod_type and value is not None:
            try:
                # Convert to float and handle NaN
                val = float(value)
                if not np.isnan(val) and val > 0:
                    type_totals[prod_type] += val
            except (ValueError, TypeError):
                continue
    
    if not type_totals:
        print("Error: No data to plot (no valid values found)")
        return
    
    # Sort by value (descending)
    sorted_types = sorted(type_totals.items(), key=lambda x: x[1], reverse=True)
    production_types = [t[0] for t in sorted_types]
    values = [t[1] for t in sorted_types]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(production_types)))
    bars = ax1.barh(production_types, values, color=colors)
    ax1.set_xlabel('Installed Capacity (MW)', fontsize=12, fontweight='bold')
    ax1.set_title('Installed Capacities by Production Type', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        if not np.isnan(val) and val > 0:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{val:,.0f} MW', 
                    ha='left', va='center', fontweight='bold')
    
    # Pie chart - filter out zero/NaN values
    valid_indices = [i for i, v in enumerate(values) if not np.isnan(v) and v > 0]
    if valid_indices:
        pie_values = [values[i] for i in valid_indices]
        pie_labels = [production_types[i] for i in valid_indices]
        pie_colors = [colors[i] for i in valid_indices]
        ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
        ax2.set_title('Capacity Distribution by Production Type', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Capacity Distribution by Production Type', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_capacities_time_series(data: Dict[str, Any], title: str = "Capacity Over Time", 
                                save_path: Optional[str] = None, show: bool = True):
    """
    Plot time series of capacities if date range data is available.
    
    Args:
        data: API response with time series data
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Try to extract time series data from different possible structures
    time_series_data = {}
    
    # Check for different possible data structures
    if "capacities_per_production_type" in data:
        entries = data["capacities_per_production_type"]
    elif "capacities_per_production_unit" in data:
        entries = data["capacities_per_production_unit"]
    elif "capacities_cpc" in data:
        entries = data["capacities_cpc"]
    else:
        print("Error: Could not find capacity data in response")
        return
    
    # Group by production type and date
    for entry in entries:
        prod_type = entry.get("production_type", "Unknown")
        value = entry.get("value", 0)
        date_str = entry.get("updated_date") or entry.get("start_date") or entry.get("date")
        
        if date_str:
            try:
                # Parse ISO 8601 date
                date = pd.to_datetime(date_str)
                if prod_type not in time_series_data:
                    time_series_data[prod_type] = []
                time_series_data[prod_type].append((date, value))
            except:
                continue
    
    if not time_series_data:
        print("No time series data found. Try using date range parameters.")
        return
    
    # Sort by date for each production type
    for prod_type in time_series_data:
        time_series_data[prod_type].sort(key=lambda x: x[0])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(time_series_data)))
    
    for (prod_type, values), color in zip(time_series_data.items(), colors):
        dates = [v[0] for v in values]
        capacities = [v[1] for v in values]
        ax.plot(dates, capacities, marker='o', label=prod_type, linewidth=2, color=color, markersize=6)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Installed Capacity (MW)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_production_units(data: Dict[str, Any], top_n: int = 20, save_path: Optional[str] = None, show: bool = True):
    """
    Plot top production units by capacity.
    
    Args:
        data: API response from get_capacities_per_production_unit
        top_n: Number of top units to display
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    if "capacities_per_production_unit" not in data:
        print("Error: Invalid data structure")
        return
    
    units = data["capacities_per_production_unit"]
    
    # Extract unit information
    unit_data = []
    for entry in units:
        unit_name = entry.get("production_unit_name", entry.get("name", "Unknown"))
        value = entry.get("value", 0)
        prod_type = entry.get("production_type", "Unknown")
        unit_data.append({
            "name": unit_name,
            "capacity": value,
            "type": prod_type
        })
    
    if not unit_data:
        print("Error: No data to plot")
        return
    
    # Sort by capacity and take top N
    unit_data.sort(key=lambda x: x["capacity"], reverse=True)
    top_units = unit_data[:top_n]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(top_units) * 0.4)))
    
    names = [u["name"] for u in top_units]
    capacities = [u["capacity"] for u in top_units]
    types = [u["type"] for u in top_units]
    
    # Color by production type
    unique_types = list(set(types))
    color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    type_colors = {t: color_map[i] for i, t in enumerate(unique_types)}
    colors = [type_colors[t] for t in types]
    
    bars = ax.barh(names, capacities, color=colors)
    ax.set_xlabel('Installed Capacity (MW)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Production Units by Installed Capacity', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, capacities):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f} MW', 
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_capacity_comparison(data_list: List[Dict[str, Any]], labels: List[str], 
                            save_path: Optional[str] = None, show: bool = True):
    """
    Compare capacities across different data sources or time periods.
    
    Args:
        data_list: List of API responses to compare
        labels: Labels for each dataset
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    if len(data_list) != len(labels):
        print("Error: Number of datasets must match number of labels")
        return
    
    # Extract all production types
    all_types = set()
    for data in data_list:
        if "capacities_per_production_type" in data:
            for entry in data["capacities_per_production_type"]:
                if "production_type" in entry:
                    all_types.add(entry["production_type"])
    
    if not all_types:
        print("Error: No data to plot")
        return
    
    all_types = sorted(list(all_types))
    
    # Build comparison matrix
    comparison_data = {label: {} for label in labels}
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if "capacities_per_production_type" in data:
            for entry in data["capacities_per_production_type"]:
                prod_type = entry.get("production_type")
                value = entry.get("value", 0)
                if prod_type:
                    comparison_data[label][prod_type] = value
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(all_types))
    width = 0.8 / len(labels)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [comparison_data[label].get(pt, 0) for pt in all_types]
        offset = (i - len(labels)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Production Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Installed Capacity (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Capacity Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_renewable_vs_nonrenewable(data: Dict[str, Any], save_path: Optional[str] = None, show: bool = True):
    """
    Plot comparison between renewable and non-renewable energy sources.
    
    Args:
        data: API response from get_capacities_per_production_type
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    if "capacities_per_production_type" not in data:
        print("Error: Invalid data structure")
        return
    
    # Define renewable energy types (matching actual API response types)
    renewable_types = {
        'BIOMASS', 'GEOTHERMAL', 'HYDRO_PUMPED_STORAGE', 
        'HYDRO_RUN_OF_RIVER_AND_POUNDAGE', 'HYDRO_WATER_RESERVOIR',
        'MARINE', 'SOLAR', 'WIND_OFFSHORE', 'WIND_ONSHORE', 'OTHER_RENEWABLE',
        # Also include simplified types from actual API responses
        'HYDRO', 'WIND', 'BIOGAZ'
    }
    
    # Aggregate values by production type first
    type_totals = defaultdict(float)
    
    for entry in data["capacities_per_production_type"]:
        prod_type = entry.get("production_type")
        value = entry.get("value")
        
        if prod_type and value is not None:
            try:
                val = float(value)
                if not np.isnan(val) and val > 0:
                    type_totals[prod_type] += val
            except (ValueError, TypeError):
                continue
    
    if not type_totals:
        print("Error: No valid data to plot")
        return
    
    renewable_total = 0
    nonrenewable_total = 0
    renewable_breakdown = {}
    nonrenewable_breakdown = {}
    
    for prod_type, total_value in type_totals.items():
        if prod_type in renewable_types:
            renewable_total += total_value
            renewable_breakdown[prod_type] = renewable_breakdown.get(prod_type, 0) + total_value
        else:
            nonrenewable_total += total_value
            nonrenewable_breakdown[prod_type] = nonrenewable_breakdown.get(prod_type, 0) + total_value
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall pie chart - only if we have valid totals
    if renewable_total > 0 or nonrenewable_total > 0:
        sizes = [renewable_total, nonrenewable_total]
        # Filter out zeros
        sizes_filtered = [s for s in sizes if s > 0]
        labels_filtered = ['Renewable', 'Non-Renewable']
        labels_filtered = [labels_filtered[i] for i, s in enumerate(sizes) if s > 0]
        colors_overall = ['#2ecc71', '#e74c3c']
        colors_filtered = [colors_overall[i] for i, s in enumerate(sizes) if s > 0]
        
        if sizes_filtered:
            ax1.pie(sizes_filtered, labels=labels_filtered, autopct='%1.1f%%', 
                   startangle=90, colors=colors_filtered)
        ax1.set_title('Renewable vs Non-Renewable Capacity', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Renewable vs Non-Renewable Capacity', fontsize=12, fontweight='bold')
    
    # Renewable breakdown
    if renewable_breakdown:
        renewable_types_list = list(renewable_breakdown.keys())
        renewable_values = [v for v in renewable_breakdown.values() if not np.isnan(v) and v > 0]
        renewable_types_list = [renewable_types_list[i] for i, v in enumerate(renewable_breakdown.values()) 
                               if not np.isnan(v) and v > 0]
        
        if renewable_values:
            colors_renew = plt.cm.Greens(np.linspace(0.4, 0.9, len(renewable_values)))
            ax2.pie(renewable_values, labels=renewable_types_list, autopct='%1.1f%%', 
                   startangle=90, colors=colors_renew)
        ax2.set_title('Renewable Energy Breakdown', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No renewable data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Renewable Energy Breakdown', fontsize=12, fontweight='bold')
    
    # Non-renewable breakdown
    if nonrenewable_breakdown:
        nonrenewable_types_list = list(nonrenewable_breakdown.keys())
        nonrenewable_values = [v for v in nonrenewable_breakdown.values() if not np.isnan(v) and v > 0]
        nonrenewable_types_list = [nonrenewable_types_list[i] for i, v in enumerate(nonrenewable_breakdown.values()) 
                                  if not np.isnan(v) and v > 0]
        
        if nonrenewable_values:
            colors_nonrenew = plt.cm.Reds(np.linspace(0.4, 0.9, len(nonrenewable_values)))
            ax3.pie(nonrenewable_values, labels=nonrenewable_types_list, autopct='%1.1f%%', 
                   startangle=90, colors=colors_nonrenew)
        ax3.set_title('Non-Renewable Energy Breakdown', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No non-renewable data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Non-Renewable Energy Breakdown', fontsize=12, fontweight='bold')
    
    # Bar comparison
    categories = ['Renewable', 'Non-Renewable']
    totals = [renewable_total if not np.isnan(renewable_total) else 0, 
              nonrenewable_total if not np.isnan(nonrenewable_total) else 0]
    colors_overall = ['#2ecc71', '#e74c3c']
    bars = ax4.bar(categories, totals, color=colors_overall, alpha=0.8)
    ax4.set_ylabel('Installed Capacity (MW)', fontsize=12, fontweight='bold')
    ax4.set_title('Total Capacity Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, totals):
        if not np.isnan(val) and val > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,.0f} MW', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    dotenv.load_dotenv()

    client_id = os.getenv("RTE_CLIENT")
    client_secret = os.getenv("RTE_SECRET")

    if not client_id or not client_secret:
        print("Warning: RTE_CLIENT and RTE_SECRET not found in .env file. API calls will be skipped.")
        print("Using local data instead if available.")
    else:
        rte = RTEAPI(client_id, client_secret)

        # Example 1: Get capacities per production type (no date range)
        print("=== Example 1: Capacities per Production Type ===")
        try:
            data1 = rte.get_capacities_per_production_type()
            print(data1)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "="*50 + "\n")

        # Example 2: Get capacities per production type with date range
        print("=== Example 2: Capacities per Production Type (with date range) ===")
        try:
            # Get data for a specific period (e.g., last 30 days)
            # Note: API requires ISO 8601 format with timezone (e.g., '2023-02-01T00:00:00+01:00')
            # For France, timezone is typically +01:00 (CET) or +02:00 (CEST)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Format with timezone (using +01:00 for CET, adjust as needed)
            start_str = start_date.strftime("%Y-%m-%dT00:00:00+01:00")
            end_str = end_date.strftime("%Y-%m-%dT00:00:00+01:00")
            
            data2 = rte.get_capacities_per_production_type(
                start_date=start_str,
                end_date=end_str
            )
            print(data2)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "="*50 + "\n")

        # Example 3: Get capacities per production unit
        print("=== Example 3: Capacities per Production Unit ===")
        try:
            data3 = rte.get_capacities_per_production_unit()
            print(data3)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "="*50 + "\n")

        # Example 4: Get CPC capacities
        print("=== Example 4: CPC Capacities ===")
        try:
            data4 = rte.get_capacities_cpc()
            print(data4)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "="*50 + "\n")

        # Example 5: Visualizations
        print("=== Example 5: Creating Visualizations ===")
        try:
            # Get data for visualization
            viz_data = rte.get_capacities_per_production_type()
            
            # Plot 1: Capacities per production type
            print("Creating production type visualization...")
            plot_capacities_per_production_type(viz_data, save_path="capacities_by_type.png", show=True)
            
            # Plot 2: Renewable vs Non-renewable
            print("Creating renewable vs non-renewable comparison...")
            plot_renewable_vs_nonrenewable(viz_data, save_path="renewable_comparison.png", show=True)
            
            # Plot 3: Production units (if available)
            try:
                units_data = rte.get_capacities_per_production_unit()
                print("Creating production units visualization...")
                plot_production_units(units_data, top_n=15, save_path="top_production_units.png", show=True)
            except Exception as e:
                print(f"Could not plot production units: {e}")
            
            # Plot 4: Time series (if date range is used)
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                start_str = start_date.strftime("%Y-%m-%dT00:00:00+01:00")
                end_str = end_date.strftime("%Y-%m-%dT00:00:00+01:00")
                
                time_series_data = rte.get_capacities_per_production_type(
                    start_date=start_str,
                    end_date=end_str
                )
                print("Creating time series visualization...")
                plot_capacities_time_series(time_series_data, 
                                           title="Installed Capacity Evolution Over Time",
                                           save_path="capacity_time_series.png", 
                                           show=True)
            except Exception as e:
                print(f"Could not plot time series: {e}")
                
        except Exception as e:
            print(f"Error creating visualizations: {e}")

