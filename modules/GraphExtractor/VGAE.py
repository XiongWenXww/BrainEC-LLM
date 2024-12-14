import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class VGAE(nn.Module):
	def __init__(self, args, adj):
		super(VGAE,self).__init__()
		self.VAGE_in_dim = args.VAGE_in_dim # T
		self.device = args.device
		self.base_gcn = GraphConvSparse(args, args.VAGE_in_dim, args.VAGE_hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args, args.VAGE_hidden1_dim, args.VAGE_in_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args, args.VAGE_hidden1_dim, args.VAGE_in_dim, adj, activation=lambda x:x)
		self.activation = nn.ELU()
		self.dropout = nn.Dropout(p=args.dropout)
		self.layer_norm = nn.LayerNorm(args.ts_len).to(args.device)

	def encode(self, X):
		hidden, _ = self.base_gcn(X)
		self.mean, adj_mu = self.gcn_mean(hidden) # [B, T, N]
		self.logstd, adj_logstd = self.gcn_logstddev(hidden)
		# print('mean:', self.mean.shape)
		gaussian_noise = torch.randn(self.VAGE_in_dim, X.size(2)).to(self.device) # [T, N]
		# print('gaussian:', gaussian_noise.shape, 'torch.exp(self.logstd)):', torch.exp(self.logstd).shape)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean # [T, N]dot_product[B, T, N]
		gaussian_noise = torch.randn(X.size(2), X.size(2)).to(self.device) # [N, N]
		adj = gaussian_noise*torch.exp(adj_logstd) + adj_mu
		return sampled_z, adj

	def forward(self, X):
		Z, adj = self.encode(X) # [B, T, N], [N, N]
		# adj = dot_product_decode(Z.transpose(1, 2)) # [B, N, N]
		# Z:[B, T, N]
		# print('VGAE adj:', adj)

		Z = self.dropout(self.activation(Z))
		adj = self.dropout(self.activation(adj))
		adj = F.softmax(adj, dim=-1) # 使得行和为1
		

		# Z += X
		# Z = self.layer_norm(Z.transpose(1, 2)).transpose(1, 2) # [B, T, N]
		return Z, adj
	
# change the second dim
class GraphConvSparse(nn.Module):
	def __init__(self, args, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim).to(args.device)
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		self.adj = self.adj + torch.eye(self.adj.shape[0]).to(self.adj.device)
		D = torch.diag(self.adj.sum(axis=1))
		D_inv_sqrt = torch.diag(torch.pow(D.sum(dim=1), -0.5))
		# D_inv_sqrt = np.linalg.inv(np.sqrt(D))
		self.adj = D_inv_sqrt @ self.adj @ D_inv_sqrt # D^{-1/2}AD^{-1/2}
		x = inputs # [B, T, N]
		# print(self.weight.transpose(0, 1).shape, x.shape)
		x = torch.matmul(self.weight.transpose(0, 1), x) # [d_out, T]×[B, T, N] = [B, d_out, N]
		# print(x.shape)
		x = torch.matmul(x, self.adj)# [B, d_out, N]×[N, N] = [B, d_out, N]
		# x = torch.matmul(x, self.weight)
		# x = torch.matmul(self.adj, x)
		outputs = self.activation(x)
		return outputs, self.adj # [B, d_out, N], [N, N]


def dot_product_decode(Z):
	# [B, T, N]
	A_pred = torch.sigmoid(torch.matmul(Z,Z.permute(0, 2, 1)))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


# class GAE(nn.Module):
# 	def __init__(self,args, adj):
# 		super(GAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args, args.VAGE_in_dim, args.VAGE_hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args, args.VAGE_hidden1_dim, args.VAGE_hidden2_dim, adj, activation=lambda x:x)

# 	def encode(self, X):
# 		hidden = self.base_gcn(X)
# 		z = self.mean = self.gcn_mean(hidden)
# 		return z

# 	def forward(self, X):
# 		Z = self.encode(X)
# 		A_pred = dot_product_decode(Z)
# 		return A_pred
		

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args, args.VAGE_in_dim, args.VAGE_hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args, args.VAGE_hidden1_dim, args.VAGE_hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args, args.VAGE_hidden1_dim, args.VAGE_hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out