import torch

@torch.jit.script
def check_waypoints(robot,waypts,idx):
    """
    Robot: [nx2]
    Waypts: [nx2]
    idx: [m]
    Waypts[idx]: [mx2]

    Returns True where target needs to be updated to next waypoint, False otherwise
    """
    m = - (waypts[idx+1][:,0] - waypts[idx-1][:,0]) / (waypts[idx+1][:,1] - waypts[idx-1][:,1] + 1e-7)
    b = - m * waypts[idx][:,0] + waypts[idx][:,1]

    R = m * robot[:,0] + b - robot[:,1]
    W = m * waypts[idx-1][:,0] + b - waypts[idx-1][:,1]

    #return torch.where(torch.sign(R) == torch.sign(W),0,1) #increment if the robot is past the waypoint
    return torch.sign(R) != torch.sign(W) #increment if the robot is past the waypoint

if __name__ == '__main__':
    waypts = torch.tensor([[0,0],[0,0.5],[1,1],[0.5,1.5],[0.5,2],[1,2],[1.5,2],[2,2]])
    idx = torch.tensor([1,2,3,4])
    robot = torch.tensor([[.1,.1],[1,1],[1,1],[1,1]])
    c = check_waypoints(robot,waypts,idx)
    print(c)
