<img width="780" height="253" alt="ACM w color" src="https://github.com/user-attachments/assets/4e2708f9-6273-472f-98f8-48993d924992" />

This script "RETURN DRIVERS USING PRINCIPAL COMPONENT ANALYSIS" performs the following tasks:

Fetch Stock Price Data: It uses the yfinance library to download historical stock price data for a list of tickers from the Johannesburg Stock Exchange (JSE).
The data is fetched for the date range starting from January 1, 2025, to the current date.

Compute Returns: The script calculates daily percentage returns from the stock price data.

Perform Principal Component Analysis (PCA): It applies PCA to the computed returns to identify the main factors (principal components) driving the returns. The analysis is limited to the top 3 components.

The script is designed to analyze and reduce the dimensionality of stock return data, which can be useful for identifying patterns or factors influencing stock performance.

