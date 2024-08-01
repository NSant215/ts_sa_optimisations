# ts_sa_optimisations
Analysis on using the Tabu Search and Simulated Annealing practical optimisation algorithms on a complex minimisation problem. The result of these investigations are reported in Report.pdf.

This project explores the performance of the Simulated Annealing (SA) and Tabu Search (TS) algorithms in minimising the 6D Eggholder function. This problem is hard to solve as we have many local maxima and minima, with a global optimum on a constraint boundary. Our problem is made easier by the fact all variables are continuous and on the same scale, all constraints are inequalities and our feasible region is not disjoint. We used the 2D Eggholder function (d=2) first in order to visualise how the algorithm is operating and also varied the parameters of each algorithm to see what what they have on the performance.

## Project Organisation
There are helper functions written in eggholder.py and init_archive.py to hold code for rendering the 2d eggholder plots and storing archive points. The two python notebooks then contain code specific to that algorithm with the various investigations that were carried out as detailed in the report.
