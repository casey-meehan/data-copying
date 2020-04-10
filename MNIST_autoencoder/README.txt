All files in the './perceptual_similarity' directory are borrowed from Richard Zhang et al. as a product of their work 
"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
(https://arxiv.org/abs/1801.03924). Specifically, this is borrowed from Richard Zhang's companion
git repository: https://github.com/richzhang/PerceptualSimilarity. We made use of these tools to 
learn a latent space embedding with perceptually meaningful L2 distance. We additionally hope that
-- since this Zhang et al's perceptual distance is not trained on MNIST -- our embedding is not overfit
to the MNIST training set. 
