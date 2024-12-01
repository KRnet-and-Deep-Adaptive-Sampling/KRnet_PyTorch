### The standard PyTorch implementation of KRnet

The source code includes the main file *train_KRnet_pdf.py*, a directory *models* for the definition of KRnet. The current *train_KRnet_pdf.py* is for PDF approximation, assuming that the unscaled probability density function is given instead of data. 

For the details of the structure of KRnet and its application to unscaled PDF approximation, please refer to

*X. Wan and S. Wei, VAE-KRnet and its applications to variational Bayes, Communications in Computational Physics, 31 (2022), pp. 1049-1082.*

*K. Tang, X. Wan and Q. Liao, Adaptive deep density approximation for Fokker-Planck equations, Journal of Computational Physics, 457 (2022), 111080.*

The implementation of the rotation layer is from *Junjie He*. You can find his original GitHub repository here: [Junjie He's implementation](https://github.com/CS-He/torchKRnet)