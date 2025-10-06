import torch

def create_nd_correlated_data(n_samples, n_dim, r=0.7):
    n_half = n_dim // 2
    mean = torch.zeros(n_dim)
    
    cov = torch.eye(n_dim)
    
    for i in range(n_dim):
        cov[i, n_dim - i - 1] = r

    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    dep_data = mvn.sample((n_samples,))
    
    A = dep_data[:, :n_half]
    B = dep_data[:, n_half:]
    
    indep_data = torch.cat([A, B[torch.randperm(n_samples)]], dim=1)
    
    return dep_data, indep_data, cov


def random_covariance_matrix(n):
    A = torch.randn(n, n)
    cov = A @ A.T
    d = torch.sqrt(torch.diag(cov))
    cov = cov / (d.unsqueeze(0) * d.unsqueeze(1))
    return cov


def create_randomly_correlated_normal_data(n_samples, n_dim):
    n_half = n_dim // 2
    mean = torch.zeros(n_dim)
    
    cov = random_covariance_matrix(n_dim)

    joint = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    dep_data = joint.sample((n_samples,))
    
    cov_1 = cov[:n_half, :n_half]
    cov_2 = cov[n_half:, n_half:]
    
    marginal_1 = torch.distributions.MultivariateNormal(torch.zeros(n_half), covariance_matrix=cov_1)
    marginal_2 = torch.distributions.MultivariateNormal(torch.zeros(n_half), covariance_matrix=cov_2)
    
    A = marginal_1.sample((n_samples,))
    B = marginal_2.sample((n_samples,))
    
    indep_data = torch.cat([A, B], dim=1)
    
    return dep_data, indep_data, cov    


def create_block_correlated_data(n_samples, n_dim, r=0.7):
    n_half = n_dim // 2
    
    mean = torch.zeros(n_half)
    cov_1 = torch.eye(n_half)
    
    for i in range(n_half):
        cov_1[i, n_half - i - 1] = r

    marginal = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_1)
    indep_data = torch.cat([marginal.sample((n_samples,)), marginal.sample((n_samples,))], dim=1)
    
    mean = torch.zeros(n_dim)
    cov = torch.block_diag(cov_1, cov_1)
    
    joint = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    dep_data = joint.sample((n_samples,))
    
    return dep_data, indep_data, cov


def MI_multivariate_normal(cov):
    n_dim = cov.shape[1]
    n_half = n_dim // 2
    cov_1 = cov[:n_half, :n_half]
    cov_2 = cov[n_half:, n_half:]
    det_cov_1 = torch.abs(torch.linalg.det(cov_1))
    det_cov_2 = torch.abs(torch.linalg.det(cov_2))
    det_cov = torch.abs(torch.linalg.det(cov))
    return torch.log(det_cov_1 * det_cov_2 / det_cov) / 2
