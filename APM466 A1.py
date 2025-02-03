#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from datetime import datetime
from scipy.optimize import newton, brentq, fsolve
from scipy.interpolate import interp1d
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt


# # Read in Data

# In[71]:


# Read in Data
long_bonds = pd.read_csv('BondPrices(1).csv', skiprows=1, nrows=19) # Split 3-10 year bonds
short_bonds = pd.read_csv('BondPrices(1).csv', skiprows=23) # Split 0-3 year bonds

# Clean Data and Set Consistent Format
for df in [long_bonds, short_bonds]:
    df['Issue Date'] = pd.to_datetime(df['Issue Date'])
    df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
    df['Coupon'] = df['Coupon'].astype(float)
    price_cols = [col for col in df.columns if 'Jan' in col]
    df[price_cols] = df[price_cols].replace('#VALUE!', np.nan).astype(float) # Replace with null values
    
# Set index to ISIN values
long_bonds.set_index('ISIN', inplace=True)
short_bonds.set_index('ISIN', inplace=True)

print("Long bonds shape:", long_bonds.shape)
print("Short bonds shape:", short_bonds.shape)
print("Short bonds first few rows:\n", short_bonds.head())


# # Ten Bond Selection

# In[72]:


ten_bonds = [
    'CA135087K528', 'CA135087K940', 'CA135087L518', 'CA135087L930',
    'CA135087M847', 'CA135087N837', 'CA135087P576', 'CA135087Q491',
    'CA135087Q988', 'CA135087R895'
]

# Maturities of selected bonds
selected_bonds = pd.concat([long_bonds, short_bonds]).loc[ten_bonds]
print("Selected 10 bond maturities:")
for isin in ten_bonds:
    maturity = selected_bonds.loc[isin, 'Maturity Date']
    print(f"{isin}: {maturity.strftime('%Y-%m-%d')}")


# # Yield Curve

# In[73]:


def calc_dirty_price(clean_price, coupon, prev_payment, calc_date):
    """Convert clean price to dirty price by adding accrued interest."""
    days_since_payment = (calc_date - prev_payment).days
    accrued_interest = (days_since_payment / 365) * coupon * 100
    return clean_price + accrued_interest

def calc_ytm(price, par, coupon, maturity_date, calc_date, freq=2):
    """Calculate yield to maturity using actual day count"""
    if price <= 0:
        return np.nan

    # Calculate periods and cash flow dates
    periods = 0
    coupon_date = maturity_date
    while calc_date < coupon_date:
        coupon_date = coupon_date - pd.DateOffset(months=6)
        periods += 1
    
    # Semi-annual coupon payment
    coupon_payment = (coupon / freq) * par
    
    # Generate cash flows and payment dates
    cash_flows = [coupon_payment] * (periods - 1) + [coupon_payment + par]
    payment_dates = [coupon_date + (i+1)*pd.DateOffset(months=6) for i in range(periods)]
    
    # Calculate dirty price
    dirty_price = calc_dirty_price(price, coupon, coupon_date, calc_date)
    
    def ytm_equation(y):
        """Present value equation using actual days"""
        return sum([cf / (1 + y)**((date - calc_date).days / 365) 
                   for cf, date in zip(cash_flows, payment_dates)]) - dirty_price
    
    ytm_solution = fsolve(ytm_equation, x0=0.05)[0]
    return max(ytm_solution, 0)

def calculate_yield_curves(selected_bonds, ref_date=datetime(2025, 1, 6)):    
    price_dates = [col for col in selected_bonds.columns if 'Jan' in col]
    ytm_data = []

    for date in price_dates:
        ytms = {'Date': datetime.strptime(date, '%b %d').strftime('%b %d')}
        calc_date = datetime.strptime(f"2025 {date}", '%Y %b %d')
        
        for isin, bond in selected_bonds.iterrows():
            price = bond[date]
            if pd.isna(price) or price <= 0:
                continue
                
            maturity_date = bond['Maturity Date']
            if maturity_date <= calc_date:
                continue
                
            ytm = calc_ytm(price, 100, bond['Coupon'], maturity_date, calc_date)
            years_to_maturity = (maturity_date - calc_date).days / 365
            
            if not np.isnan(ytm):
                ytms[f"{int(years_to_maturity)}Y"] = round(ytm * 100, 4)
        
        if len(ytms) > 1:
            ytm_data.append(ytms)
            
    return pd.DataFrame(ytm_data).fillna('')

def plot_yield_curves(ytm_df):
    """Plot yield curves"""
    plt.figure(figsize=(12, 8))
    
    for _, row in ytm_df.iterrows():
        date = row['Date']
        maturities = [int(col[:-1]) for col in row.index if 'Y' in col]
        yields = [row[f"{m}Y"] for m in maturities]
        plt.plot(maturities, yields, marker='o', label=date)

    plt.title('Government of Canada Yield Curves')
    plt.xlabel('Years to Maturity')
    plt.ylabel('Yield to Maturity (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()


# In[ ]:


# def calc_dirty_price(clean_price, coupon, prev_payment, calc_date):
#     """Convert clean price to dirty price by adding accrued interest."""
#     days_since_payment = (calc_date - prev_payment).days
#     accrued_interest = (days_since_payment / 365) * coupon * 100
#     return clean_price + accrued_interest

# def calc_ytm(price, par, coupon, maturity_date, calc_date, freq=2):
#     """Calculate yield to maturity using actual day count."""
#     if price <= 0:
#         return np.nan

#     # Calculate periods and cash flow dates
#     periods = 0
#     coupon_date = maturity_date
#     while calc_date < coupon_date:
#         coupon_date = coupon_date - pd.DateOffset(months=6)
#         periods += 1
    
#     # Semi-annual coupon payment
#     coupon_payment = (coupon / freq) * par
    
#     # Generate cash flows and payment dates
#     cash_flows = [coupon_payment] * (periods - 1) + [coupon_payment + par]
#     payment_dates = [coupon_date + (i+1)*pd.DateOffset(months=6) for i in range(periods)]
    
#     # Calculate dirty price
#     dirty_price = calc_dirty_price(price, coupon, coupon_date, calc_date)
    
#     def ytm_equation(y):
#         """Present value equation using actual days."""
#         return sum([cf / (1 + y)**((date - calc_date).days / 365) 
#                    for cf, date in zip(cash_flows, payment_dates)]) - dirty_price
    
#     ytm_solution = fsolve(ytm_equation, x0=0.05)[0]
#     return max(ytm_solution, 0)

# def calculate_interpolated_yield_curves(selected_bonds, ref_date=datetime(2025, 1, 6)):
#     """Calculate yield curves using linear interpolation for standard intervals."""
#     price_dates = [col for col in selected_bonds.columns if 'Jan' in col]
#     ytm_data = []

#     for date in price_dates:
#         ytms = {'Date': datetime.strptime(date, '%b %d').strftime('%b %d')}
#         calc_date = datetime.strptime(f"2025 {date}", '%Y %b %d')
        
#         # Calculate YTM and exact maturity for each bond
#         maturities = []
#         rates = []
        
#         for isin, bond in selected_bonds.iterrows():
#             price = bond[date]
#             if pd.isna(price) or price <= 0:
#                 continue
                
#             maturity_date = bond['Maturity Date']
#             if maturity_date <= calc_date:
#                 continue
                
#             ytm = calc_ytm(price, 100, bond['Coupon'], maturity_date, calc_date)
#             years_to_maturity = (maturity_date - calc_date).days / 365
            
#             if not np.isnan(ytm):
#                 maturities.append(years_to_maturity)
#                 rates.append(ytm)
        
#         # Create interpolation function
#         if maturities and rates:
#             f = interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
            
#             # Calculate rates at standard intervals
#             for year in range(1, 6):
#                 ytms[f"{year}Y"] = round(float(f(year)) * 100, 4)
        
#         ytm_data.append(ytms)
            
#     return pd.DataFrame(ytm_data).fillna('')

# # Calculate and display interpolated YTMs
# ytm_df = calculate_interpolated_yield_curves(selected_bonds)

# # Print results
# print("\nInterpolated YTM curves data:")
# print(ytm_df.to_string(index=False))

# # Print original YTMs for one date for comparison
# print("\nOriginal YTMs for Jan 6 (uninterpolated):")
# calc_date = datetime(2025, 1, 6)
# sorted_bonds = selected_bonds.sort_values('Maturity Date')
# for isin, bond in sorted_bonds.iterrows():
#     if bond['Maturity Date'] > calc_date:
#         ytm = calc_ytm(bond['Jan 6'], 100, bond['Coupon'], bond['Maturity Date'], calc_date)
#         years = (bond['Maturity Date'] - calc_date).days / 365
#         print(f"Years: {years:.4f}, YTM: {ytm*100:.4f}%")


# In[74]:


# Compute yield curves using already-selected bonds
ytm_df = calculate_yield_curves(selected_bonds)

# Plot yield curves
plot_yield_curves(ytm_df)

# Print formatted yield curve data
print("Yield curves data:")
print(ytm_df.to_string(index=False))


# In[78]:


# Create a DataFrame with ISIN index and dates as columns
ytm_detail_df = pd.DataFrame(index=selected_bonds.index)

# Calculate YTM for each bond and date
for date in [col for col in selected_bonds.columns if 'Jan' in col]:
    calc_date = datetime.strptime(f"2025 {date}", '%Y %b %d')
    ytms = []
    
    for isin, bond in selected_bonds.iterrows():
        price = bond[date]
        maturity_date = bond['Maturity Date']
        
        if pd.isna(price) or price <= 0 or maturity_date <= calc_date:
            ytms.append(np.nan)
        else:
            ytm = calc_ytm(price, 100, bond['Coupon'], maturity_date, calc_date)
            ytms.append(ytm * 100)  # Convert to percentage
            
    ytm_detail_df[date] = ytms

# Add maturity date and coupon columns for reference
ytm_detail_df['Maturity'] = selected_bonds['Maturity Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
ytm_detail_df['Coupon'] = selected_bonds['Coupon']

# Sort by maturity date
ytm_detail_df = ytm_detail_df.sort_values('Maturity')

# Print results
pd.set_option('display.float_format', '{:.4f}'.format)
print("\nYield to Maturity (%) for each bond and date:")
print(ytm_detail_df.to_string())
pd.reset_option('display.float_format')


# # Bootstrap Spot Curve

# In[79]:


def find_dirty_price(price, coupon, maturity_date, calc_date):
    """Calculate dirty price by adding accrued interest"""
    last_coupon = maturity_date
    while calc_date < last_coupon:
        last_coupon = last_coupon - pd.DateOffset(months=6)
    
    days_since_last_coupon = (calc_date - last_coupon).days
    dirty_price = price + (days_since_last_coupon / 365) * coupon * 100
    return dirty_price

def bootstrap_spot_rates(bonds_df, calc_date, price_col):
    """
    Bootstrap spot rates from bond prices using iterative approach
    Returns list of spot rates corresponding to each bond's maturity
    """
    spot_rates = []
    bonds_df = bonds_df.sort_values('Maturity Date')
    
    for _, bond in bonds_df.iterrows():
        clean_price = bond[price_col]
        if pd.isna(clean_price):
            continue
            
        coupon = bond['Coupon']
        maturity = bond['Maturity Date']
        
        # Calculate dirty price
        dirty_price = find_dirty_price(clean_price, coupon, maturity, calc_date)
        
        # Adjust coupon to semi-annual payments
        semi_annual_coupon = coupon / 2 * 100
        
        # Calculate coupon dates and number of periods
        periods = 0
        coupon_date = maturity
        while calc_date < coupon_date:
            coupon_date = coupon_date - pd.DateOffset(months=6)
            periods += 1
            
        # Generate cash flow dates
        cash_flow_dates = [coupon_date + (i+1)*pd.DateOffset(months=6) for i in range(periods)]
        cash_flows = [semi_annual_coupon] * (periods - 1) + [semi_annual_coupon + 100]
        
        if len(spot_rates) == 0:
            # For first bond, solve directly
            time_to_maturity = (cash_flow_dates[0] - calc_date).days / 365
            spot_rate = ((100 + semi_annual_coupon) / dirty_price) ** (1 / time_to_maturity) - 1
            spot_rates.append(spot_rate)
        else:
            # Use previous spot rates to discount interim cash flows
            discounted_coupons = 0
            for t in range(min(periods-1, len(spot_rates))):
                time_to_cf = (cash_flow_dates[t] - calc_date).days / 365
                discounted_coupons += cash_flows[t] / (1 + spot_rates[t])**time_to_cf
            
            # Solve for the new spot rate
            time_to_maturity = (cash_flow_dates[periods-1] - calc_date).days / 365
            spot_rate = ((100 + semi_annual_coupon) / (dirty_price - discounted_coupons)) ** (1 / time_to_maturity) - 1
            spot_rates.append(spot_rate)
    
    return spot_rates

def calculate_spot_curves(selected_bonds):
    """Calculate spot curves for each date"""
    price_dates = [col for col in selected_bonds.columns if 'Jan' in col]
    spot_data = []
    
    for date_col in price_dates:
        calc_date = datetime.strptime(f"2025 {date_col}", '%Y %b %d')
        spots = {'Date': date_col}
        
        # Get spot rates for all bonds on this date
        daily_spot_rates = bootstrap_spot_rates(selected_bonds, calc_date, date_col)
        
        if daily_spot_rates:
            # Map spot rates to years based on time to maturity
            sorted_bonds = selected_bonds.sort_values('Maturity Date').reset_index(drop=True)
            for idx in range(len(sorted_bonds)):
                bond = sorted_bonds.iloc[idx]
                time_to_maturity = (bond['Maturity Date'] - calc_date).days / 365
                year = int(round(time_to_maturity))
                if 0 < year <= 5 and idx < len(daily_spot_rates):  # Only include rates for 1-5 years
                    spots[f"{year}Y"] = round(daily_spot_rates[idx] * 100, 4)
        
        spot_data.append(spots)
    
    return pd.DataFrame(spot_data)


def plot_spot_curve(spot_df):
    """Plot spot rate curves"""
    plt.figure(figsize=(10, 6))
    for _, row in spot_df.iterrows():
        date = row['Date']
        maturities = [int(col[:-1]) for col in row.index if 'Y' in col]
        rates = [row[col] for col in row.index if 'Y' in col]
        plt.plot(maturities, rates, marker='o', label=date)
    plt.xlabel("Years to Maturity")
    plt.ylabel("Spot Rate (%)")
    plt.title("Spot Rate Curve")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.show()


# In[80]:


# Calculate and display the results
spot_df = calculate_spot_curves(selected_bonds)
print("\nSpot curve data:")
print(spot_df.to_string(index=False))

# Plot the curves
plot_spot_curve(spot_df)


# In[81]:


def format_spot_rates_detail(selected_bonds, calc_date_str):
    """Format spot rates with linearly interpolated standard intervals"""
    calc_date = datetime.strptime(f"2025 {calc_date_str}", '%Y %b %d')
    
    # Sort bonds by maturity for bootstrapping
    sorted_bonds = selected_bonds.sort_values('Maturity Date')
    spot_rates = bootstrap_spot_rates(sorted_bonds, calc_date, calc_date_str)
    
    # Calculate exact maturity years
    maturity_years = [(bond['Maturity Date'] - calc_date).days / 365.0 
                     for _, bond in sorted_bonds.iterrows()]
    
    # Create linear interpolation function
    f = interp1d(maturity_years, spot_rates, kind='linear', fill_value='extrapolate')
    
    # Standard intervals from 0.5 to 5.0 years, stepping by 0.5
    standard_years = np.arange(0.5, 5.1, 0.5)
    interpolated_rates = f(standard_years)
    
    # Create DataFrame with the required format
    spot_data = {
        'date': [calc_date.strftime('%Y-%m-%d')] * len(standard_years),
        'maturity_years': standard_years,
        'spot': interpolated_rates
    }
    
    spot_df = pd.DataFrame(spot_data)
    return spot_df

# Calculate and display spot rates for Jan 6
jan6_spots = format_spot_rates_detail(selected_bonds, 'Jan 6')

# Set display options for clean output
pd.set_option('display.float_format', lambda x: '%.6f' % x)
print("\nLinearly interpolated spot rates at standard intervals:")
print(jan6_spots.to_string(index=True))

# Also print original values for comparison
print("\nOriginal spot rates before interpolation:")
calc_date = datetime(2025, 1, 6)
for isin, bond in selected_bonds.sort_values('Maturity Date').iterrows():
    years = (bond['Maturity Date'] - calc_date).days / 365.0
    price_col = 'Jan 6'
    spot_rate = bootstrap_spot_rates(selected_bonds, calc_date, price_col)[
        list(selected_bonds.index).index(isin)
    ]
    print(f"Years: {years:.4f}, Rate: {spot_rate:.6f}")


# # Forward Rate

# In[82]:


def calculate_forward_rates(spot_df):
    """Calculate forward rates from spot rates"""
    forward_data = []
    
    for _, row in spot_df.iterrows():
        forwards = {'Date': row['Date']}
        
        # Calculate forward rates for 1yr to n-years
        for n in range(2, 6):
            if f"{n}Y" in row and "1Y" in row:
                spot_n = row[f"{n}Y"] / 100
                spot_1 = row["1Y"] / 100
                
                # Calculate forward rate using semi-annual compounding
                forward = ((1 + spot_n)**(2*n) / (1 + spot_1)**(2))**(1/(2*(n-1))) - 1
                forwards[f"1Y-{n-1}Y"] = round(forward * 100, 4)
        
        forward_data.append(forwards)
    
    return pd.DataFrame(forward_data)

def plot_forward_curve(forward_df):
    """Plot forward rate curves"""
    plt.figure(figsize=(10, 6))
    for _, row in forward_df.iterrows():
        date = row['Date']
        maturities = [int(col.split('-')[1][:-1]) for col in row.index if '1Y-' in col]
        rates = [row[col] for col in row.index if '1Y-' in col]
        plt.plot(maturities, rates, marker='o', label=date)
    plt.xlabel("Forward Period (Years)")
    plt.ylabel("Forward Rate (%)")
    plt.title("Forward Rate Curve")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.show()


# In[83]:


# Calculate and display the results
forward_df = calculate_forward_rates(spot_df)

print("\nForward curve data:")
print(forward_df.to_string(index=False))

# Plot the curves
plot_forward_curve(forward_df)


# In[77]:


def format_forward_rates_detail(spot_df, calc_date_str='Jan 6'):
    """Format forward rates showing forward years and maturity years"""
    # Calculate forward rates from spot rates
    date_spot_rates = spot_df[spot_df['Date'] == calc_date_str].iloc[0]
    
    forward_data = {
        'date': [],
        'forward_years': [],
        'maturity_years': [],
        'forward_rate': []
    }
    
    # Calculate 1-year forward rates for 2,3,4,5 year maturities
    for n in range(2, 6):
        if f"{n}Y" in date_spot_rates and "1Y" in date_spot_rates:
            spot_n = date_spot_rates[f"{n}Y"] / 100
            spot_1 = date_spot_rates["1Y"] / 100
            
            # Calculate forward rate using the relationship between spot rates
            forward = ((1 + spot_n)**(n) / (1 + spot_1))**(1/(n-1)) - 1
            
            forward_data['date'].append('2025-01-06')
            forward_data['forward_years'].append(1)
            forward_data['maturity_years'].append(n)
            forward_data['forward_rate'].append(forward)
    
    forward_df = pd.DataFrame(forward_data)
    return forward_df

# Calculate and display forward rates
forward_detail = format_forward_rates_detail(spot_df)

# Set display options for clean output
pd.set_option('display.float_format', lambda x: '%.6f' % x)
print("\nForward rates with years detail:")
print(forward_detail.to_string(index=True))

# Reset display options
pd.reset_option('display.float_format')


# # Covariance Matrices

# In[58]:


def calculate_log_returns_and_cov(rates_df, rate_type="yield"):
    """Calculate log returns and covariance matrix"""
    # Select appropriate columns based on rate type
    if rate_type == "yield":
        rate_cols = [f"{i}Y" for i in range(1, 6)]  # 1Y through 5Y
    else:
        rate_cols = [f"1Y-{i}Y" for i in range(1, 5)]  # 1Y-1Y through 1Y-4Y
    
    # Calculate log returns: Xi,j = log(ri,j+1/ri,j)
    log_returns = pd.DataFrame()
    for col in rate_cols:
        if col in rates_df.columns:
            values = rates_df[col].values / 100  # Convert from percentage to decimal
            log_returns[col] = np.log(values[1:] / values[:-1])
    
    # Calculate covariance matrix
    cov_matrix = log_returns.cov()
    
    return log_returns, cov_matrix

# Calculate covariance matrices for both yield and forward rates
# For yield rates (5x5 matrix)
yield_returns, yield_cov = calculate_log_returns_and_cov(spot_df, "yield")
print("\nYield rates covariance matrix:")
print(yield_cov.to_string())

# For forward rates (4x4 matrix)
forward_returns, forward_cov = calculate_log_returns_and_cov(forward_df, "forward")
print("\nForward rates covariance matrix:")
print(forward_cov.to_string())


# # Eigenvalues and Eigenvectors

# In[88]:


def calculate_log_returns_and_cov(rates_df, rate_type="yield"):
    """Calculate log returns and covariance matrix"""
    if rate_type == "yield":
        rate_cols = [f"{i}Y" for i in range(1, 6)]  # 1Y through 5Y
    else:
        rate_cols = [f"1Y-{i}Y" for i in range(1, 5)]  # 1Y-1Y through 1Y-4Y
    
    log_returns = pd.DataFrame()
    for col in rate_cols:
        if col in rates_df.columns:
            values = rates_df[col].values
            log_returns[col] = np.log(values[1:] / values[:-1])
    
    return log_returns.cov()

def analyze_eigenvalues(cov_matrix):
    """Calculate and analyze eigenvalues and eigenvectors of covariance matrix"""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure first eigenvector components are positive
    if eigenvectors[0, 0] < 0:
        eigenvectors[:, 0] *= -1
    
    # Normalize each eigenvector
    for i in range(eigenvectors.shape[1]):
        norm = np.linalg.norm(eigenvectors[:, i])
        eigenvectors[:, i] = eigenvectors[:, i] / norm
        
    return eigenvalues, eigenvectors

# Calculate for yield rates
yield_cov = calculate_log_returns_and_cov(spot_df, "yield")
yield_eigenvals, yield_eigenvecs = analyze_eigenvalues(yield_cov)
print("Yield Covariance Matrix:")
print(pd.DataFrame(yield_cov).to_string())
print("\nYield Eigenvalues:")
print(yield_eigenvals)
print("\nYield Eigenvectors:")
print(yield_eigenvecs)

# Calculate for forward rates
forward_cov = calculate_log_returns_and_cov(forward_df, "forward")
forward_eigenvals, forward_eigenvecs = analyze_eigenvalues(forward_cov)
print("\nForward Covariance Matrix:")
print(pd.DataFrame(forward_cov).to_string())
print("\nForward Eigenvalues:")
print(forward_eigenvals)
print("\nForward Eigenvectors:")
print(forward_eigenvecs)


# In[ ]:




