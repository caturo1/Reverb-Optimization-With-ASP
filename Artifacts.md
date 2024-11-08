
| Type                                                                               | Desc                                                                                                                                                             | Correlation                                                                                  |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Flutter                                                                            | Grainy, uneven amplitude modulation of resulting IR                                                                                                              | echo density/ diffusion                                                                      |
| Excessive Resonance/ Ringing/ Modal Bunching                                       | Distinct emphasis on certain resonances. Might measure difference between spectral peaks before and after application to detect excessive ringing and coloration | Diffusion, Feedback, Filtering (peak clustering)                                             |
| Frequency Masking                                                                  |                                                                                                                                                                  | rTime                                                                                        |
| Transient clarity/overloading/                                                     | Self-Explanatory                                                                                                                                                 | preDelay, early reflactions (measure if attack will be preserved/ compare transients/onsets) |
| Undifferentiated sound                                                             | Too much $T_{60}$ on signals with high energy in low frequencies                                                                                                 | rTime/diffusion                                                                              |
| Phase Cencellation(though very hard to detect and gives the reverb also character) | Hollow sound due to destructive interferences                                                                                                                    | Phase Correlation between channels                                                           |
| Low End Stacking                                                                   | Lower frequencies persist in reverb tail                                                                                                                         | Damping                                                                                      |

Questions:
1. What is the ACTUAL research question?
   ASP in audio effect processing, how do we create a knowledge base, which constraints do we use, performance testing (file length etc.), frequency distribution
2. Agree on a reverb architecture
3. Shall I implement the reverb as well, since ADE seems to be keen of that and this would give me better control of specific parameters
4. How did you implement the ASP guessing as a real time component
5. General signal path
6. Next meeting have a running prototype und then write the paper

