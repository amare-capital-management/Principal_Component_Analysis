"""
RETURN DRIVERS USING PRINCIPAL COMPONENT ANALYSIS 
Fetches stock price data, computes returns, performs PCA, and saves factor returns and exposures.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------------
# List of tickers
# -------------------------------
tickers = ["ABG.JO", "ADH.JO", "AEL.JO", "AFE.JO", "AFH.JO", "AFT.JO", "AGL.JO", "AHR.JO", "AIP.JO", "ANG.JO", "ANH.JO", "APN.JO", "ARI.JO",
          "ARL.JO", "ATT.JO", "AVI.JO", "BAW.JO", "BHG.JO", "BID.JO", "BLU.JO", "BOX.JO", "BTI.JO", "BTN.JO", "BVT.JO", "BYI.JO", "CFR.JO", "CLS.JO",
          "CML.JO", "COH.JO", "CPI.JO", "CSB.JO", "DCP.JO", "DRD.JO", "DSY.JO", "DTC.JO", "EMI.JO", "EQU.JO", "EXX.JO", "FBR.JO", "FFB.JO", "FSR.JO",
          "FTB.JO", "GFI.JO", "GLN.JO", "GND.JO", "GRT.JO", "HAR.JO", "HCI.JO", "HDC.JO", "HMN.JO", "HYP.JO", "IMP.JO", "INL.JO", "INP.JO", "ITE.JO",
          "JSE.JO", "KAP.JO", "KIO.JO", "KRO.JO", "KST.JO", "LHC.JO", "LTE.JO", "MCG.JO", "MKR.JO", "MNP.JO", "MRP.JO", "MSP.JO", "MTH.JO", "MTM.JO",
          "MTN.JO", "N91.JO", "NED.JO", "NPH.JO", "NPN.JO", "NRP.JO", "NTC.JO", "NY1.JO", "OCE.JO", "OMN.JO", "OMU.JO", "OUT.JO", "PAN.JO", "PHP.JO",
          "PIK.JO", "PMR.JO", "PPC.JO", "PPH.JO", "PRX.JO", "QLT.JO", "RBX.JO", "RCL.JO", "RDF.JO", "REM.JO", "RES.JO", "RLO.JO", "RNI.JO", "S32.JO",
          "SAC.JO", "SAP.JO", "SBK.JO", "SHC.JO", "SHP.JO", "SLM.JO", "SNT.JO", "SOL.JO", "SPG.JO", "SPP.JO", "SRE.JO", "SRI.JO", "SSS.JO",
          "SSU.JO", "SSW.JO", "SUI.JO", "TBS.JO", "TFG.JO", "TGA.JO", "TKG.JO", "TRU.JO", "TSG.JO", "VAL.JO", "VKE.JO", "VOD.JO", "WBC.JO", "WHL.JO"]
# -----------------------------
# -------------------------------
# Fetch price data
# -------------------------------
start_date = "2025-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

data = yf.download(tickers, start=start_date, end=end_date)['Close']

data = data.dropna(axis=1, how='all')

print("Tickers successfully downloaded:", data.columns.tolist())

# -------------------------------
# Compute returns
# -------------------------------
returns = data.pct_change().dropna()

if returns.shape[0] == 0:
    raise ValueError("No returns data available. Check your tickers and date range.")

# -------------------------------
# Perform PCA
# -------------------------------
pca = PCA(n_components=3)
pca.fit(returns)

pct = pca.explained_variance_ratio_
pca_components = pca.components_
cum_pct = np.cumsum(pct)

# -------------------------------
# Save PCA contribution plot
# -------------------------------
x = np.arange(1, len(pct) + 1, 1)
fig1, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(x, pct * 100, align="center")
axes[0].set_title("Contribution (%)")
axes[0].set_xticks(x)
axes[0].set_xlim([0, 4])
axes[0].set_ylim([0, 100])

axes[1].plot(x, cum_pct * 100, "ro-")
axes[1].set_title("Cumulative contribution (%)")
axes[1].set_xticks(x)
axes[1].set_xlim([0, 4])
axes[1].set_ylim([0, 100])

plt.tight_layout()
fig1.savefig("pca_contribution_plots.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

# -------------------------------
# Compute factor returns
# -------------------------------
X = np.asarray(returns)
factor_returns = X.dot(pca_components.T)
factor_returns = pd.DataFrame(factor_returns,
                              columns=["f1", "f2", "f3"],
                              index=returns.index)

factor_returns.to_csv(f"factor_returns.csv")

# -------------------------------
# Compute factor exposures
# -------------------------------
factor_exposures = pd.DataFrame(pca_components.T,
                                index=returns.columns,
                                columns=["f1", "f2", "f3"])

factor_exposures.to_csv(f"factor_exposures.csv")


print("PCA analysis complete! CSVs and plot saved.")

