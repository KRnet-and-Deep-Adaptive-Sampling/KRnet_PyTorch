import torch
import numpy as np
import argparse
import time
# from BR_lib_He import *
import models.BR_model as BR_model

from icecream import ic
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 


seed = 909
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_default_dtype(torch.float32)


class KRnet4pdf(torch.nn.Module):
    def __init__(self, name, device, args):
        super(KRnet4pdf, self).__init__()

        self.name = name
        self.device = device
        self.args = args


        self.vae_prior = BR_model.IM_rNVP_KR_CDF(n_dim=args.n_dim_V,
                                                 n_step=args.n_step_F,
                                                 n_depth=args.n_depth_F_prior,
                                                 n_width=args.n_width_F,
                                                 n_bins=args.n_bins_F,
                                                 rotation=args.rotation_F)
        

    def actnorm_data_initialization(self):
        # if self.vae_encoder is not None:
        #     self.vae_encoder.actnorm_data_initialization()
        # if self.vae_decoder is not None:
        #     self.vae_decoder.actnorm_data_initialization()
        if self.vae_prior is not None:
            self.vae_prior.actnorm_data_initialization()
        # if self.vae_pz_post is not None:
        #     self.vae_pz_post.actnorm_data_initialization()
    

    def forward(self, inputs):
        z = inputs
        # ic(z)
        x = self.vae_prior.mapping_from_prior(z)
        # x = x.detach()

        # ic(x.abs().mean())

        log_pyx_k_1 = self.log_pdf_prior(x)


        # ic(log_pyx_k_1)

        loss = log_pyx_k_1

        # ic(self.log_unscaled_pdf(x))
        loss -= self.log_unscaled_pdf(x)

        return torch.mean(loss)

    def log_pdf_prior(self, inputs):
        x = inputs
        log_pdf = self.vae_prior(x)

        return log_pdf
    
    def log_unscaled_pdf(self, x):

        d = x.shape[1]  # 维度d
    
        # 根据d定义系数数组c
        c = 2 * torch.ones(d).to(self.device)
        c[-2] = 7
        c[-1] = 200

        # 计算v(x)
        ci_sq_xi_sq = c[:-1]**2 * x[:, :-1]**2
        ci1_xi1 = c[1:] * x[:, 1:]
        term = ci_sq_xi_sq + (ci1_xi1 + 5 * (ci_sq_xi_sq + 1))**2
        v_x = torch.sum(term, dim=1, keepdims=True)
        return -v_x/2




def main(args):

    # Computing device
    device = torch.device('cuda:0')
    ic(device)


    def create_model():
        # build up the model
        pdf_model = KRnet4pdf('KRnet_for_pdf', device, args)
        return pdf_model


    y_train = torch.randn(args.n_train, args.n_dim_V)
    data_flow = TensorDataset(y_train)
    train_dataset = DataLoader(data_flow, shuffle=True, batch_size=args.batch_size)

    it = iter(train_dataset)
    y_init,  = next(it)
    # ic(y_init.shape)
    y_init = y_init.to(device)

    pdf_model = create_model().to(device)
    # pdf_model.set_linear_inv_prb(A, y_hat, sigma, P, C, mu0)
    # pdf_model.actnorm_data_initialization()
    ic(pdf_model(y_init).item())

    # stochastic gradient method ADAM
    optimizer = torch.optim.AdamW(pdf_model.parameters(), lr=args.lr)


    loss_vs_epoch=[]
    KL_vs_epoch=[]

    loss_global_min = np.inf

    n_epochs = args.n_epochs
    iteration = 0
    
    for i in range(1,n_epochs+1):
        # freeze the rotation layers after a certain number of epochs

        start_time = time.time()
        # iterate over the batches of the dataset
        for step, train_batch in enumerate(train_dataset):
            
            optimizer.zero_grad()

            y_batch,  = train_batch
            y_batch = y_batch.to(device)
            loss = pdf_model(y_batch)

            loss.backward()
            optimizer.step()

            for name, param in pdf_model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                            print(f"Gradient of {name} has NaNs or Infs.")
                    else:
                        print(f"Gradient of {name} is None.")


            iteration += 1
        
        elapsed = time.time() - start_time

            #print('epoch %s, iteration %s, loss = %s' %
            #             (i, iteration, loss_metric.result().numpy()))
            # write the summary file
            #if tf.equal(optimizer.iterations % args.log_step, 0):
            #    tf.summary.scalar('loss', loss_metric.result(), step=optimizer.iterations)

        ic(i, iteration, loss.item(), loss_global_min, elapsed)

        if loss.item() < loss_global_min:
            loss_global_min = loss.item()
            if args.n_collect_stat == 1 and i >= args.n_collect_epoch:
                # j = i - args.n_collect_epoch
                if i % args.n_draw_samples == 0:
                    visulization = pdf_model.vae_prior.mapping_from_prior(y_batch)
                    visulization = visulization.detach().clone().cpu().numpy()
                    np.savetxt('./Rosenbrock_KRnet_{}.dat'.format(i), visulization)


    # np.savetxt('pdf_approx_mean.dat', mean_global_min.numpy())
    # np.savetxt('pdf_approx_var.dat', var_global_min.numpy())


if __name__ == '__main__':

    desc = "KRnet4pdf"
    p = argparse.ArgumentParser(description=desc)

    p.add_argument('--data_dir', type=str, help='Path to preprocessed data files.')

    # save parameters
    p.add_argument('--ckpts_dir', type=str, default='./pdf_ckpt', help='Path to the check points.')
    p.add_argument('--summary_dir', type=str, default='./pdf_summary', help='Path to the summaries.')
    p.add_argument('--log_step', type=int, default=16, help='Record information every n optimization iterations.')
    p.add_argument('--ckpt_step', type=int, default=100, help='Save the model every n epochs.')

    p.add_argument("--dimension_reduction", type=int, default=1, help='Dimension reduction or not. 0: No; 1: Yes.')
    p.add_argument("--full_model", type=int, default=0, help='Full model. 0: KRnet; 1: Gaussian iid')

    p.add_argument("--n_dim_F", type=int, default=4, help='The number of random dimensions for latent space.')
    p.add_argument("--n_dim_V", type=int, default=32, help='The number of random dimensions of data')

    # Neural network hyperparameters for flow-based generative model.
    p.add_argument('--n_depth_F_prior', type=int, default=6, help='The number of affine coupling layers.')
    p.add_argument('--n_depth_F_post', type=int, default=2, help='The number of affine coupling layers for the posterior.')
    p.add_argument('--n_width_F', type=int, default=32, help='The number of neurons for the hidden layers.')
    p.add_argument('--n_step_F', type=int, default=4, help='The step size for dimension reduction in each squeezing layer.')
    p.add_argument('--shrink_rate_F', type=float, default=1.0, help='The shrinking rate of the width of NN.')
    p.add_argument('--rotation_F', action='store_true', help='Specify rotation layers or not?')
    p.set_defaults(rotation_F=True)
    p.add_argument('--n_bins_F', type=int, default=32, help='The number of bins for uniform partition of the support of PDF.')
    p.add_argument('--lbda', type=float, default=-1, help='The penalty parameter for the mutual information.')


    # Neural network hyperparameteris for VAE.
    p.add_argument('--n_depth_V_en', type=int, default=2, help='The number of hidden layers for VAE encoder.')
    p.add_argument('--n_width_V_en', type=int, default=32, help='The number of neurons for the hidden layers in VAE encoder.')
    p.add_argument('--n_depth_V_de', type=int, default=2, help='The number of hidden layers for VAE decoder.')
    p.add_argument('--n_width_V_de', type=int, default=32, help='The number of neurons for the hidden layers in VAE decoder.')

    #optimization hyperparams:
    p.add_argument("--n_train", type=int, default=100000, help='The number of samples in the training set.')
    p.add_argument('--batch_size', type=int, default=100000, help='Batch size of training generator.')
    p.add_argument("--lr", type=float, default=0.001, help='Base learning rate.')
    p.add_argument('--n_epochs',type=int, default=25000, help='Total number of training epochs.')

    # samples:
    p.add_argument("--n_samples", type=int, default=100000, help='Sample size for the trained model.')
    p.add_argument("--n_collect_stat", type=int, default=1, help='Collect statistics or not. 1: yes; 0: no')
    p.add_argument("--n_collect_epoch", type=int, default=5000, help='Collect data starting with this epoch number')
    p.add_argument("--n_draw_samples", type=int, default=500, help='Draw samples every n epochs.')

    args = p.parse_args()

    main(args)