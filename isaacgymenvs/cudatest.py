import torch                                                                                                   
torch.backends.cuda.matmul.allow_tf32 = False                                                                  
torch.backends.cudnn.benchmark = True                                                                          
torch.backends.cudnn.deterministic = False                                                                     
torch.backends.cudnn.allow_tf32 = True                                                                         
data = torch.randn([512, 1, 240, 240], dtype=torch.float, device='cuda',requires_grad=True).to(memory_format=torch.channels_last)                                                                          
net = torch.nn.Conv2d(1, 8, kernel_size=[16, 16], padding=[0, 0], stride=[4, 4], dilation=[1, 1], groups=1)    
net = net.cuda().float().to(memory_format=torch.channels_last)                                                 
out = net(data)                                                                                                
out.backward(torch.randn_like(out))                                                                            
torch.cuda.synchronize() 
