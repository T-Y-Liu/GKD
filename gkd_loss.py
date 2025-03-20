class GraphConvolution(nn.Module):
    def __init__(self, input_features, output_features):
        super(GraphConvolution, self).__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(36, 1)).cuda() 
        # self.weight2 = nn.Parameter(torch.FloatTensor(input_features, 1)).cuda() 
        self.weight2 = nn.Parameter(torch.FloatTensor(36, 1)).cuda() 
        self.reset_parameters()
        self.input_features = input_features
    
    def reset_parameters(self):
        stdv = 1. / self.weight1.size(1)**0.5
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / self.weight2.size(1)**0.5
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, input, start, end, boundary):
        # print(self.input_features)
        input  = input.cuda() 
        start = start.cuda()
        end = end.cuda()
        student_neighbours = torch.einsum('ed, bcdf -> bcef', (start, input[:boundary]))
        updated_student_neighbours = input[boundary:] + torch.einsum('ed, bcdf -> bcef', (end.t(), self.weight1 * student_neighbours))
        teacher_neighbours = torch.einsum('ed, bcdf -> bcef', (start, input[boundary:]))
        updated_teacher_neighbours = input[:boundary] + torch.einsum('ed, bcdf -> bcef', (end.t(), self.weight2 * teacher_neighbours))

        return torch.cat((updated_student_neighbours, updated_teacher_neighbours), dim=0)

def create_adj_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end


def gnn_loss(feat_student, feat_teacher, T):
    b, c, d, w, h = feat_student.shape
    
    feat_student = feat_student.view(b, c, d, w*h)
    feat_teacher = feat_teacher.view(b, c, d, w*h)
    gc = GraphConvolution(w*h, w*h)  
    feat_cat = torch.cat((feat_student, feat_teacher), dim=0)
    # print(feat_cat.size())
    start, end = create_adj_matrix(d)
    # feat_student = feat_student.view(b, c, d, w*h).transpose(1, 2)
    # feat_teacher = feat_teacher.view(b, c, d, w*h).transpose(1, 2)
    # gc = GraphConvolution(w*h, w*h)  
    # feat_cat = torch.cat((feat_student, feat_teacher), dim=0)
    # print("feat_studen",feat_student.size())
    # start, end = create_adj_matrix(c)

    
    
    gc_result = gc(feat_cat, start, end, b)

    _b,_c,_d,_hw = gc_result[:b].size()
    # 为什么做softmax

    pred1 = F.softmax(gc_result[:b].view(_b, _c, _d, int(np.sqrt(_hw)), int(np.sqrt(_hw))), dim=-1)
    pred2 = F.softmax(gc_result[b:].view(_b, _c, _d, int(np.sqrt(_hw)), int(np.sqrt(_hw))), dim=-1)

    # 计算KL散度

    kl_div = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(pred1/T, dim=1),
                                F.softmax(pred2/T, dim=1)) * T * T 
    # mse_loss = nn.MSELoss(reduction='mean')(F.softmax(pred1/T, dim=1),
    #                                     F.softmax(pred2/T, dim=1)) * T * T
    # cos_sim = F.cosine_similarity(pred1, pred2, dim=1)
    return kl_div