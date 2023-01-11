import streamlit as st
import pandas as pd
import numpy as np
from numpy import exp
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
sns.set_style("whitegrid")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



def nonlin(rs, time, ksw, sorbent_mass): 
    # create the full equation for the non/curvilinear phase for a passive sampler with limited WBL intereference
    return ksw * sorbent_mass * (1 - exp(-(rs * time)/(ksw * sorbent_mass)))

# estimate sampling rate for the curvilinear phase for a passive sampler with limited WBL intereference
# this uses the nonlin function and a curve fitting method from scipy.optimize
def two_phase_nonlin_fit(time_column, compound_column, time_unit = 'day', water_unit = 'mL'): 
    # use the scipy optimise curve fit function to fit the data to the nonlin equation, estimating unknown parameters
    params, covs = curve_fit(nonlin, time_column, compound_column)
    
    # pulls out the estimated sorbent water partitioning coefficient (ksw) and the estimated sampling rate
    ksw, sampling_rate = params[0], params[1]
    
    #st.write(str(ksw), str(sampling_rate + water_unit + "/" + time_unit))
    # plot_range creates a range from the lowest to highest time point, allowing the curve fit to plot smoothly by interpolating unknowns between data points
    plot_range = np.arange(min(time_column), max(time_column))
    # substantiate the graph space
    fig, ax = plt.subplots()
    ax.plot(time_column, compound_column, 'ko', label="y-original")
    ax.plot(plot_range, nonlin(plot_range, *params), label="a*M*(1-exp(-(r*x)/(a*M)))", color = 'black')
    plt.xlabel('Day')
    plt.ylabel('ng per sampler / ng per mL water')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    expander2.pyplot(fig)
    #return ksw, sampling_rate



st.title("Passive sampling calculators")

expander1 = st.expander("Kinetic sampling rate (no flow)")
expander2 = st.expander("Curvilinear sampling rate (no flow)")
expander3 = st.expander("Time weighted average (kinetic)")
expander4 = st.expander("Equilibrium concentration")


no_flow_data = expander1.file_uploader("Upload your csv with time in column 1 and each chemical's ns/cw in subsequent columns (chemical name as row 1)")

if no_flow_data != None:
    no_flow_data = pd.read_csv(no_flow_data)

    no_flow_button = expander1.button("show dataframe")
    if no_flow_button:
        df = expander1.dataframe(no_flow_data)

    else:
        expander1.write("")

    no_flow_calc_button = expander1.button("Calculate sampling rate")
    if no_flow_calc_button:
        result = sp.stats.linregress(no_flow_data.iloc[:, 0], no_flow_data.iloc[:, 1])
        expander1.write("Estimated kinetic mode sampling rate (slope) = " + str(round(result.slope,3)) + " volume/time")
        expander1.write(u"R\u00B2 = " + str(round(result.rvalue**2, 3)))
        expander1.write(u"p-value = " + str(round(result.pvalue,5)))
        expander1.write("Note: if p-value is far below 0.05 it will display 0.0")
    else:
        expander1.write("")
    kinetic_plot = expander1.button("Graph estimated sampling rate")
    if kinetic_plot:
        lin_reg = sns.lmplot(data = no_flow_data, x = no_flow_data.columns[0], y =  no_flow_data.columns[1], line_kws = {'color': 'black'}, scatter_kws = {'color' : 'black'})
        lin_reg.set(ylim=0, xlim=0)
        expander1.pyplot(lin_reg)
    else:
        expander1.write("")

    


curv_data = expander2.file_uploader("Upload your curvilinear csv with time in column 1 and each chemical's ns/cw in subsequent columns (chemical name as row 1)")

if curv_data != None:
    curv_data = pd.read_csv(curv_data)

    curv_button = expander2.button("show curvilinear dataframe")
    if curv_button:
        df = expander2.dataframe(curv_data)
    else:
        expander2.write("")

    curv_calc_button = expander2.button("Calculate curvilinear sampling rate")
    if curv_calc_button:
        two_phase_nonlin_fit(curv_data.iloc[:,0], curv_data.iloc[:,1])




rs = expander3.number_input('Sampling rate:')
twa_data = expander3.file_uploader("Upload your csv with row 1 as chemical name, row 2 as sampling rate, row 3 as concentration of analyte in sampler and row 4 as time deployed")
# cw = (N/Rs*t)
# N is the amount of the chemical accumulated by the sampler (typically ng),
# Rs is the sampling rate (L/d), and
# t is the exposure time (d). 


ksw = expander4.number_input("Sorbent water partitioning coefficient (Ksw)")
equi_data = expander4.file_uploader("Upload your csv with row 1 as chemical name, row 2 as Ksw and row 3 as mass of analyte in sampler")