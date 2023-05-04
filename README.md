# CS-229---Foundations-of-Deep-Learning

## Understanding and Mitigating Extrapolation Failures in Physics-Informed Neural Networks

- Luca D'Amico-Wong, Lukas Fesser, and Richard Qiu

### Individual Contributions:
- **Luca D'Amico-Wong**:
    - Implemented Fourier analysis for all PDEs as described in section 4
    - Implemented Wasserstein-Fourier distance metric to analyze shfits in spectra and attempted to use these to predict L2 error (Cazelles+2020)
    - Conducted additional experiments on training dynamics and extrapolation behavior
    - Wrote midterm report for project
    
- **Lukas Fesser**:
    - Came up with the original project idea
	- Implement the code for the experiments in section 3
	- Suggested investigating the Fourier spectra of different PINN solutions
	- Suggested and implemented the experiments on changes in the amplitude in section 4

    
- **Richard Qiu**:
    - Implemented PyTorch version of original PINN paper (Raissi+2017)
    - Implemented DPM based on Kim+2020 
    - Implemented transfer learning based on Desai+2021
    - Ran experiments for DPM and transfer learning

### Contents of this repo:

- **deepxde_experiments**: contains the code for all the equations and experiments in sections 3 and 4. The individual notebooks contain comments/ instructions how to run them.
- **fourier_analysis**: contains all the code for section 4 and Appendix sections A1 and A2.
- **pytorch_implementation**: contains all code for the DPM method discussed in section 2.3 and for the transfer learning experiments in section 5.
- **final_report.pdf**: the final report for this project in the NeurIPS 2023 format.
