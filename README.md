# Bond Analysis Tool

## Overview
This Python script performs comprehensive analysis of Government of Canada bonds, including yield curve calculations, spot rate bootstrapping, forward rate derivation, and risk analysis through covariance matrices and eigenvalue decomposition.

## Features
- Yield curve calculation and visualization
- Spot rate bootstrapping with linear interpolation
- Forward rate calculation
- Covariance matrix analysis for both yield and forward rates
- Eigenvalue and eigenvector analysis for risk decomposition

## Dependencies
- pandas
- numpy
- matplotlib
- scipy
- datetime

## Input Data Format
The script expects a CSV file with the following columns:
- ISIN
- Issue Date
- Maturity Date
- Coupon
- Price columns for different dates (e.g., "Jan 6", "Jan 7", etc.)

## Key Functions

### Yield Calculations
- `calc_dirty_price()`: Converts clean price to dirty price
- `calc_ytm()`: Calculates yield to maturity using actual day count
- `calculate_yield_curves()`: Generates yield curves for multiple dates

### Spot Rate Analysis
- `bootstrap_spot_rates()`: Implements spot rate bootstrapping
- `calculate_spot_curves()`: Calculates spot curves for each date
- `format_spot_rates_detail()`: Formats spot rates with linear interpolation

### Forward Rate Analysis
- `calculate_forward_rates()`: Derives forward rates from spot rates
- `plot_forward_curve()`: Visualizes forward rate curves
- `format_forward_rates_detail()`: Formats detailed forward rate information

### Risk Analysis
- `calculate_log_returns_and_cov()`: Computes covariance matrices
- `analyze_eigenvalues()`: Performs eigenvalue decomposition for risk analysis

## Usage
1. Prepare your bond data in CSV format
2. Import the required libraries
3. Load and clean the data using pandas
4. Run the desired analysis functions
5. Visualize results using the provided plotting functions

## Output
- Yield curves
- Spot rate curves
- Forward rate curves
- Covariance matrices
- Eigenvalues and eigenvectors for risk analysis

## Notes
- All calculations use actual day count convention
- Semi-annual coupon payments are assumed
- Linear interpolation is used for standard intervals
