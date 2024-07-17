print("Enter the number of inputs:")
n=int(input())
inp=[]
for i in range(n):
    inp.append(int(input()))
w1=int(input("Enter weight 1:"))
w2=int(input("Enter weight 2:"))

bias=int(input("Enter the bias:"))
lrate=float(input("Enter the learning rate:"))  
epochs=int(input("Enter the number of epochs:"))
out=[]
for i in range(n//2):  
    out.append(int(input()))

def activation_func(z):
    if z >=0:
        return 1
    else:
        return 0
    
def backpropogate(w1,w2,lrate,inp,ind,bias,z):
    w1=w1+lrate*inp[ind]
    w2=w2+lrate*inp[ind+1]
    bias=bias+lrate
    return w1, w2, bias 

def compute(n,inp,w1,w2,bias,lrate,epochs,out):
    for _ in range(epochs):
        ind=0
        while ind<n:
            z = inp[ind]*w1 + inp[ind+1]*w2 + bias
            check=activation_func(z)
            if check!=out[ind//2]: 
                w1, w2, bias = backpropogate(w1,w2,lrate,inp,ind,bias,z)
            ind+=2
    print(w1,w2,bias)

compute(n,inp,w1,w2,bias,lrate,epochs,out)