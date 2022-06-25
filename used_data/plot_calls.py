plot_usages(files=['./used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl'
    ],
     keys=[(0.00025, 0.9), 
        (0.00025, 0.7),
        (0.00025, 0.3), 
        (0.0001, 0.9), 
        (0.0001, 0.7),
        (0.0001, 0.3),
    ],
    num_x_plots=3,
    num_y_plots=2,
    figsize=(24,18),
    num_bits=3,
    titles=['compressedSGD lr=0.00025, min-max decay=0.9', 
            'compressedSGD lr=0.00025, min-max decay=0.7',
            'compressedSGD lr=0.00025, min-max decay=0.3',
            'compressedSGD lr=0.00001, min-max decay=0.9',
            'compressedSGD lr=0.00001, min-max decay=0.7',
            'compressedSGD lr=0.00001, min-max decay=0.3'
            ],
     out='used_data/plots/usages_1worker_3bits_exp.png')

plot_usages(files=['./used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl'
    ],
     keys=[(0.00025, 0.9), 
        (0.00025, 0.7),
        (0.00025, 0.3), 
        (0.0001, 0.9), 
        (0.0001, 0.7),
        (0.0001, 0.3),
    ],
    num_x_plots=3,
    num_y_plots=2,
    figsize=(24,18),
    num_bits=3,
    titles=['compressedSGD lr=0.00025, min-max decay=0.9', 
            'compressedSGD lr=0.00025, min-max decay=0.7',
            'compressedSGD lr=0.00025, min-max decay=0.3',
            'compressedSGD lr=0.0001, min-max decay=0.9',
            'compressedSGD lr=0.0001, min-max decay=0.7',
            'compressedSGD lr=0.0001, min-max decay=0.3'
            ],
     out='used_data/plots/usages_1worker_3bits_lin.png')

plot_usages(files=['./used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_zeros/histories.pkl'
    ],
     keys=[(0.0001, 0.7)
    ],
    num_x_plots=1,
    num_y_plots=1,
    figsize=(8,6),
    num_bits=3,
    titles=['compressedSGD lr=0.0001, min-max decay=0.7'
            ],
     out='used_data/plots/usages_1worker_3bits_lin_zeros.png')

plot_usages(files=['./used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_b1/histories.pkl',
    './used_data/bin_usages/compressed_sgd_vote_B64_n3_llin8_EvalInterval-2_Seed1917_Workers8_b1/histories.pkl'
    ],
     keys=[(0.0001, 1),
     (0.0001, 1)
    ],
    num_x_plots=1,
    num_y_plots=2,
    figsize=(12,9),
    num_bits=3,
    titles=['compressedSGD lr=0.0001, min-max decay=1, 1 worker',
            'compressedSGD lr=0.0001, min-max decay=1, 8 workers'
            ],
     out='used_data/plots/usages_1worker_3bits_lin_b1.png')

plot_usages(files=['./used_data/bin_usages/compressed_sgd_B64_n3_llin1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_vote_B64_n3_llin1_EvalInterval-2_Seed1917_Workers4/histories.pkl',
        './used_data/bin_usages/compressed_sgd_vote_B64_n3_llin1_EvalInterval-2_Seed1917_Workers8/histories.pkl',
        './used_data/bin_usages/compressed_sgd_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers1/histories.pkl',
        './used_data/bin_usages/compressed_sgd_vote_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers4/histories.pkl',
        './used_data/bin_usages/compressed_sgd_vote_B64_n3_lexp1_EvalInterval-2_Seed1917_Workers8/histories.pkl'
    ],
     keys=[(0.0001, 0.7), 
        (0.0001, 0.7),
        (0.0001, 0.7), 
        (0.0001, 0.7), 
        (0.0001, 0.7),
        (0.0001, 0.7),
    ],
    num_x_plots=3,
    num_y_plots=2,
    figsize=(24,18),
    num_bits=3,
    titles=['compressedSGD lr=0.0001, min-max decay=0.7, 1 worker, linear binning', 
            'compressedSGD lr=0.0001, min-max decay=0.7, 4 workers, linear binning',
            'compressedSGD lr=0.0001, min-max decay=0.7, 8 worker, linear binning',
            'compressedSGD lr=0.00001, min-max decay=0.7, 1 worker, exponential binning',
            'compressedSGD lr=0.00001, min-max decay=0.7, 4 workers, exponential binning',
            'compressedSGD lr=0.00001, min-max decay=0.7, 8 workers, exponential binning'
            ],
     out='used_data/plots/usages_different_workers.png')

plot_avg('batch_loss_hist', 
    files=['./used_data/sgd/sgd_B64_e3_EvalInterval100_Seed1917/histories.pkl',
    './used_data/sgd/sgd_B64_e3_EvalInterval100_Seed1337/histories.pkl',
    './used_data/sgd/sgd_B64_e3_EvalInterval100_Seed42/histories.pkl'
     ],
    keys=[(0.8, 1), (0.5, 1),
          (0.3, 1), (0.1, 1)
          ],
    names=['SGD lr=0.8', 'SGD lr=0.5',
           'SGD lr=0.3', 'SGD lr=0.1'],
    out='used_data/plots/SGD_lr.png',
    xaxis='number of batches',
    yaxis='(average) loss over the entire training data',
    title="SGD with different learning rates averaged over 3 Seeds",
    linspace_mul=100)

plot_avg('batch_loss_hist', 
    files=['./used_data/sign_sgd/sign_sgd_B64_e3_EvalInterval100_Seed1917/histories.pkl',
    './used_data/sign_sgd/sign_sgd_B64_e3_EvalInterval100_Seed1337/histories.pkl',
    './used_data/sign_sgd/sign_sgd_B64_e3_EvalInterval100_Seed42/histories.pkl'
     ],
    keys=[(0.001, 1), (0.0005, 1),
          (0.0003, 1), (0.0001, 1)
          ],
    names=['signSGD lr=0.001', 'signSGD lr=0.0005',
           'signSGD lr=0.0003', 'signSGD lr=0.0001'],
    out='used_data/plots/signSGD_lr.png',
    xaxis='number of batches',
    yaxis='(average) loss over the entire training data',
    title="signSGD with different learning rates averaged over 3 Seeds",
    linspace_mul=100)

plot_avg('avg_epoch_loss_hist', 
    files=['./used_data/sgd/sgd_B64_e3_EvalInterval100_Seed1917/histories.pkl',
    './used_data/sgd/sgd_B64_e3_EvalInterval100_Seed1337/histories.pkl',
    './used_data/sgd/sgd_B64_e3_EvalInterval100_Seed42/histories.pkl'
     ],
    keys=[(0.8, 1), (0.5, 1),
          (0.3, 1), (0.1, 1)
          ],
    names=['SGD lr=0.8', 'SGD lr=0.5',
           'SGD lr=0.3', 'SGD lr=0.1'],
    out='used_data/plots/epochs_SGD_lr.png',
    xaxis='epochs',
    yaxis='(average) loss over the test data',
    title="SGD with differen learning rates averaged over 3 Seeds",
    linspace_mul=1)

plot('batch_loss_hist',
    files=['./used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed42/histories.pkl',
            './used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed42/histories.pkl',
            './used_data/comparison/sign_sgd_B64_e5_EvalInterval100_Seed42/histories.pkl',
            './used_data/comparison/sign_sgd_B64_e5_EvalInterval100_Seed42/histories.pkl',
            './used_data/comparison/sgd_B64_e5_EvalInterval100_Seed42/histories.pkl',
        ],
    keys=[(0.0001, 0.7), 
            (0.0001, 0.3),
            (0.0003, 1), 
            (0.0001, 1), 
            (0.5, 1)],
    names=['compressedSGD lr=0.0001, min-max decay=0.7, linear binning, 3 bits', 'compressedSGD lr=0.0001, min-max decay=0.3, linear binning, 3 bits',
            'signSGD lr=0.0003', 'signSGD lr=0.0001',
            'SGD lr=0.5'],
    out='./used_data/plots/comparison_seed42.png',
    title='full training loss over 5 epochs for different methods',
    xaxis='number of update steps',
    yaxis='(average) loss over entire training data',
    linspace_mul=100   
)


plot('batch_loss_hist',
    files=['./used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/sign_sgd_B64_e5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/sign_sgd_B64_e5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/sgd_B64_e5_EvalInterval100_Seed1917/histories.pkl',
        ],
    keys=[(0.0001, 0.7), 
            (0.0001, 0.3),
            (0.0003, 1), 
            (0.0001, 1), 
            (0.5, 1)],
    names=['compressedSGD lr=0.0001, min-max decay=0.7, linear binning, 3 bits', 'compressedSGD lr=0.0001, min-max decay=0.3, linear binning, 3 bits',
            'signSGD lr=0.0003', 'signSGD lr=0.0001',
            'SGD lr=0.5'],
    out='./used_data/plots/comparison_seed1917.png',
    title='full training loss over 5 epochs for different methods',
    xaxis='number of update steps',
    yaxis='(average) loss over entire training data',
    linspace_mul=100   
)

plot('avg_epoch_loss_hist',
    files=['./used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/sign_sgd_B64_e5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/sign_sgd_B64_e5_EvalInterval100_Seed1917/histories.pkl',
            './used_data/comparison/sgd_B64_e5_EvalInterval100_Seed1917/histories.pkl',
        ],
    keys=[(0.0001, 0.7), 
            (0.0001, 0.3),
            (0.0003, 1), 
            (0.0001, 1), 
            (0.5, 1)],
    names=['compressedSGD lr=0.0001, min-max decay=0.7, linear binning, 3 bits', 'compressedSGD lr=0.0001, min-max decay=0.3, linear binning, 3 bits',
            'signSGD lr=0.0003', 'signSGD lr=0.0001',
            'SGD lr=0.5'],
    out='./used_data/plots/epochs_comparison_seed1917.png',
    title='test loss over 5 epochs for different methods',
    xaxis='epochs',
    yaxis='(average) loss over test data',
    linspace_mul=1  
)

plot('batch_loss_hist',
    files=['./used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917_4/histories.pkl',
        
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917_3/histories.pkl',
        
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917_2/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917_2/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_lexp5_EvalInterval100_Seed1917_3/histories.pkl'
        ],
    keys=[(0.0001, 0.7), 
            (0.0001, 0.9),
            (0.0001, 0.7),
            (0.0001, 0.3),
            
            (0.00005, 0.9),
            (0.00005, 0.7),
            (0.00005, 0.3),
            
            (0.0002, 0.9),
            (0.0002, 0.7),
            (0.0002, 0.3)
            ],
    names=['compressedSGD lr=0.0001, min-max decay=0.7, linear binning', 
            'compressedSGD lr=0.0001, min-max decay=0.9, exponential binning',
            'compressedSGD lr=0.0001, min-max decay=0.7, exponential binning',
            'compressedSGD lr=0.0001, min-max decay=0.3, exponential binning',

            'compressedSGD lr=0.00005, min-max decay=0.9, exponential binning',
            'compressedSGD lr=0.00005, min-max decay=0.7, exponential binning',
            'compressedSGD lr=0.00005, min-max decay=0.3, exponential binning',
            
            'compressedSGD lr=0.0002, min-max decay=0.9, exponential binning',
            'compressedSGD lr=0.0002, min-max decay=0.7, exponential binning',
            'compressedSGD lr=0.0002, min-max decay=0.3, exponential binning'],
    out='./used_data/plots/lin_vs_exp.png',
    title='full training loss over 5 epochs for exponential and linear binning for 3 bits',
    xaxis='number of update steps',
    yaxis='(average) loss over the full training data',
    linspace_mul=100  
)

plot('batch_loss_hist',
    files=['./used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/comparison/compressed_sgd_B64_n3_llin5_EvalInterval100_Seed1917/histories.pkl',
        
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917_2/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917_2/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917_2/histories.pkl',

        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917/histories.pkl',
        './used_data/many_bits/compressed_sgd_B64_n8_llin5_EvalInterval100_Seed1917/histories.pkl'
        ],
    keys=[(0.0001, 0.7), 
        (0.0001, 0.3),

        (0.000005, 0.9),
        (0.000005, 0.7),
        (0.000005, 0.3),  
        
        (0.0000015625, 0.9),
        (0.0000015625, 0.7),
        (0.0000015625, 0.3),
            
        (0.00001, 0.9),
        (0.00001, 0.7),
        (0.00001, 0.3)
            ],
    names=['compressedSGD lr=0.0001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 3 bits', 
            
            'compressedSGD lr=5*10^(-6), min-max decay=0.9, 8 bits',
            'compressedSGD lr=5*10^(-6), min-max decay=0.7, 8 bits',
            'compressedSGD lr=5*10^(-6), min-max decay=0.3, 8 bits',

            'compressedSGD lr=1.5625*10^(-6), min-max decay=0.9, 8 bits',
            'compressedSGD lr=1.5625*10^(-6), min-max decay=0.7, 8 bits',
            'compressedSGD lr=1.5625*10^(-6), min-max decay=0.3, 8 bits',
            
            'compressedSGD lr=0.00001, min-max decay=0.9, 8 bits',
            'compressedSGD lr=0.00001, min-max decay=0.7, 8 bits',
            'compressedSGD lr=0.00001, min-max decay=0.3, 8 bits'],
    out='./used_data/plots/3bits_vs_8bits.png',
    title='full training loss over 5 epochs for linear binning for 3 and 8 bits',
    xaxis='number of update steps',
    yaxis='(average) loss over the full training data',
    linspace_mul=100  
)

plot('batch_loss_hist',
    files=[  
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',

        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.001, 0.9),
        (0.001, 0.7), 
        (0.001, 0.3),

        (0.0001, 0.9),
        (0.0001, 0.7),
        (0.0001, 0.3),  
        
        (0.00025, 0.9),
        (0.00025, 0.7),
        (0.00025, 0.3),
            
        (0.000025, 0.9),
        (0.000025, 0.7),
        (0.000025, 0.3)
        ],
    names=[ 'compressedSGD lr=0.001, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.001, min-max decay=0.3, 3 bits', 
            
            'compressedSGD lr=0.0001, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.00025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.000025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.3, 3 bits' 


        ],
    out='./used_data/plots/decay_lr_all_3bits.png',
    title='full training loss over 7 epochs for linear binning and 3 bits',
    xaxis='number of update steps',
    yaxis='(average) loss over the full training data',
    linspace_mul=100  
)

plot('batch_loss_hist',
    files=[  
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',

        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.0001, 0.9),
        (0.0001, 0.7),
        (0.0001, 0.3),  
        
        (0.00025, 0.9),
        (0.00025, 0.7),
        (0.00025, 0.3),
            
        (0.000025, 0.9),
        (0.000025, 0.7),
        (0.000025, 0.3)
        ],
    names=['compressedSGD lr=0.0001, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.00025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.000025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.3, 3 bits' 


        ],
    out='./used_data/plots/decay_lr_focused_3bits.png',
    title='full training loss over 7 epochs for linear binning and 3 bits',
    xaxis='number of update steps',
    yaxis='(average) loss over the full training data',
    linspace_mul=100  
)

plot('avg_epoch_loss_hist',
    files=[  
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',

        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.001, 0.9),
        (0.001, 0.7), 
        (0.001, 0.3),

        (0.0001, 0.9),
        (0.0001, 0.7),
        (0.0001, 0.3),  
        
        (0.0000625, 0.9),
        (0.0000625, 0.7),
        (0.0000625, 0.3),
            
        (0.00000625, 0.9),
        (0.00000625, 0.7),
        (0.00000625, 0.3)
        ],
    names=[ 'compressedSGD lr=0.001, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.001, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.001, min-max decay=0.3, 5 bits', 
            
            'compressedSGD lr=0.0001, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.0001, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 5 bits', 

            'compressedSGD lr=0.0000625, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.0000625, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.0000625, min-max decay=0.3, 5 bits', 

            'compressedSGD lr=0.00000625, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.00000625, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.00000625, min-max decay=0.3, 5 bits' 


        ],
    out='./used_data/plots/epochs_decay_lr_all_5bits.png',
    title='test loss over 7 epochs for linear binning and 5 bits',
    xaxis='epochs',
    yaxis='(average) loss over the test data',
    linspace_mul=1  
)

plot('avg_epoch_loss_hist',
    files=[    
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',

        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.0001, 0.9),
        (0.0001, 0.7),
        (0.0001, 0.3),  
        
        (0.0000625, 0.9),
        (0.0000625, 0.7),
        (0.0000625, 0.3),
            
        (0.00000625, 0.9),
        (0.00000625, 0.7),
        (0.00000625, 0.3)
        ],
    names=[ 
            'compressedSGD lr=0.0001, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.0001, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 5 bits', 

            'compressedSGD lr=0.0000625, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.0000625, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.0000625, min-max decay=0.3, 5 bits', 

            'compressedSGD lr=0.00000625, min-max decay=0.9, 5 bits',
            'compressedSGD lr=0.00000625, min-max decay=0.7, 5 bits',
            'compressedSGD lr=0.00000625, min-max decay=0.3, 5 bits' 
        ],
    out='./used_data/plots/epochs_decay_lr_focused_5bits.png',
    title='test loss over 7 epochs for linear binning and 5 bits',
    xaxis='epochs',
    yaxis='(average) loss over the test data',
    linspace_mul=1  
)

plot('avg_epoch_loss_hist',
    files=[  
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',

        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.001, 0.9),
        (0.001, 0.7), 
        (0.001, 0.3),

        (0.0001, 0.9),
        (0.0001, 0.7),
        (0.0001, 0.3),  
        
        (0.00025, 0.9),
        (0.00025, 0.7),
        (0.00025, 0.3),
            
        (0.000025, 0.9),
        (0.000025, 0.7),
        (0.000025, 0.3)
        ],
    names=[ 'compressedSGD lr=0.001, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.001, min-max decay=0.3, 3 bits', 
            
            'compressedSGD lr=0.0001, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.00025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.000025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.3, 3 bits' 


        ],
    out='./used_data/plots/epochs_decay_lr_all_3bits.png',
    title='full training loss over 7 epochs for linear binning and 3 bits',
    xaxis='epochs',
    yaxis='(average) loss over the test data',
    linspace_mul=1 
)

plot('avg_epoch_loss_hist',
    files=[  
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',

        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.0001, 0.9),
        (0.0001, 0.7),
        (0.0001, 0.3),  
        
        (0.00025, 0.9),
        (0.00025, 0.7),
        (0.00025, 0.3),
            
        (0.000025, 0.9),
        (0.000025, 0.7),
        (0.000025, 0.3)
        ],
    names=[ 'compressedSGD lr=0.0001, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.00025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.00025, min-max decay=0.3, 3 bits', 

            'compressedSGD lr=0.000025, min-max decay=0.9, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.000025, min-max decay=0.3, 3 bits' 


        ],
    out='./used_data/plots/epochs_decay_lr_focused_3bits.png',
    title='full training loss over 7 epochs for linear binning and 3 bits',
    xaxis='epochs',
    yaxis='(average) loss over the test data',
    linspace_mul=1  
)

plot('avg_epoch_loss_hist',
    files=[  
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n3_llin_e7_EvalInterval100_Seed1917_Workers1/histories.pkl',
        './used_data/parameter_search/compressed_sgd_vote_B64_n5_llin7_EvalInterval-100_Seed1917_Workers1/histories.pkl'
        ],
    keys=[
        (0.0001, 0.7),
        (0.0001, 0.3),  
        (0.0000625, 0.9)
    ],

    names=[ 'compressedSGD lr=0.0001, min-max decay=0.7, 3 bits',
            'compressedSGD lr=0.0001, min-max decay=0.3, 3 bits',
            'compressedSGD lr=0.0000625, min-max decay=0.9, 5 bits', 
        ],
    out='./used_data/plots/epochs_5_vs_3bits.png',
    title='full training loss over 7 epochs for linear binning for 3 and 5 bits',
    xaxis='epochs',
    yaxis='(average) loss over the test data',
    linspace_mul=1  
)

