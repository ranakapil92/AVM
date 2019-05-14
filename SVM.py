# check the following link for using cvxopt qp solver
# http://cvxopt.org/examples/tutorial/qp.html


import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
import math
        
get_ipython().run_line_magic('matplotlib', 'qt')

def linear_kernel(x1, x2):   
        temp=np.dot(x1,x2)
        return temp

# the reason of adding it , because linear kernal is passed in SVM  class initialization explicitly. and class is compling  before linear_kernal 
         
    
class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Kernel/Gram matrix
       
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix((-1) * np.ones(n_samples))
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None: # linear separable case
           
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
            
        else: # soft margin case

            G = cvxopt.matrix(np.vstack(((np.diag(np.ones(n_samples) * -1)), (np.identity(n_samples)))))
            h = cvxopt.matrix(np.hstack(((np.zeros(n_samples)), (np.ones(n_samples) * self.C))))

        
        #solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Obtain the Lagrange multipliers from the solution.
        alphas = np.ravel(solution['x'])
        
        #print(alpha,alpha.shape)
        
        # Support vectors have non zero Lagrange multipliers
        # apply a threshold on the value of alpha and identify the support vectors
        # print the fraction of support vectors.
        # to do
        
        sup_vectors = alphas > 1e-3      #array of boleans, 1 for true condtions, specifying which are support vectors in alphas
          
        self.alphas = alphas[sup_vectors]  #value of alphas for support vectors
        
        self.sv = X[sup_vectors]         #suport vectors 
        self.sv_y = y[sup_vectors]       #labels of support vectors
        
        total_sv=len(self.alphas)
        
        print("%d support vectors from input %d intances, fraction=%f " % (total_sv, n_samples,float(total_sv)/float(n_samples)))
        
        
        
        # Weight vector
        # compute the weight vector using the support vectors only when using linear kernel
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(total_sv):
                self.w =self.w + (self.alphas[n] * self.sv_y[n] * self.sv[n])
        else:
            self.w = None
            
        indexs = np.arange(len(alphas))[sup_vectors]   #indexes of support vectors
            
        # Intercept
        # computer intercept term by taking the average across all support vectors
        self.b = 0
        for i in range(total_sv):
            self.b =self.b + self.sv_y[i]
            temp=0
            for j in range(total_sv):
                temp = temp + (self.alphas[j] * self.sv_y[j] * self.kernel(X[indexs[j]],X[indexs[i]]))
            self.b=self.b - temp
        self.b /= total_sv
        # to do

        

    def predict(self, X):
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        # to do
        if self.w is not None:
            temp = np.dot(X, self.w) + self.b
            return temp,np.sign(temp)
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    s += alpha * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            temp = y_predict + self.b
            return temp,np.sign(temp)
        
    

if __name__ == "__main__":
    
    
    
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        # to do
        
    def linear_kernel(x1, x2):
        temp=np.dot(x1,x2)
        return temp
        
    
    def polynomial_kernel(x1, x2, q=2):                    #change polynomonal degree here
        # implement the polynomial kernel
        # to do
        temp = (1 + np.dot(np.transpose(x1), x2)) ** q
        return temp
    
    def gaussian_kernel(x1, x2, s=2):                           #change values of s here
        # implement the radial basis function kernel
        # to do
        temp = np.exp(-linalg.norm(x1-x2)**2 / (2 * (s ** 2)))
        return temp
    
    def lin_separable_data():
        # generate linearly separable 2D data
        # remember to assign class labels as well.    
        # to do
        
        #we are using gaussain multivariate function
        
        mean1 = np.array([0, 3.5])
        mean2 = np.array([3.5, 0])
        cov = np.array([[3.5, 3], [3, 3.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        
        y1 = np.ones(100)
        y2 = (-1) * np.ones(100) 
        
        return X1, y1, X2, y2
        
    
    
    def circular_data():
        
        mean1 = [3, 3]
        noise = np.random.uniform(0,0.2,200)
        noise = noise.reshape(100,2) 
        noise = noise + mean1

        X1=np.zeros([100,2])
        X2=np.zeros([100,2])
        angle =np.random.uniform(0,1,100)*(math.pi*2)
        X1[:,0]= np.cos(angle)
        X1[:,1]= np.sin(angle)
        X1 = X1 + noise
        y1 = np.ones(100)
        
        
        angle =np.random.uniform(0,1,100)*(math.pi*2)
        X2[:,0]= np.cos(angle) + (np.cos(angle) *1)
        X2[:,1]= np.sin(angle) + (np.sin(angle) *1)
        X2 = X2 + noise
        y2 = (-1) * np.ones(100)
        
        return X1, y1, X2, y2
        
    
    def lin_separable_overlap_data():
        # for testing the soft margin implementation, 
        # generate linearly separable data where the instances of the two classes overlap.
        # to do
        
        #using the same method for generating seperable linear 2d data
        
        mean1 = np.array([0, 3.5])
        mean2 = np.array([3.5, 0])
        cov = np.array([[3.5, 1], [1, 3.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(100)
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = (-1) * np.ones(100) 
        return X1, y1, X2, y2
    
    def split_train_test(X1, y1, X2, y2):
        # split the data into train and test splits
        # to do
        X1_train = X1[:75]
        y1_train = y1[:75]
        X2_train = X2[:75]
        y2_train = y2[:75]
        X1_test = X1[75:]
        y1_test = y1[75:]
        X2_test = X2[75:]
        y2_test = y2[75:]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        
        return X_train, y_train, X_test, y_test
    
    
    
    def plot_margin(X1_train, X2_train, model):
    
        def f(x, w, b, c=0):
            y= (-w[0] * x - b + c) / w[1]     # getting value of y wrt to given paremeters of line
            return y
        
        plt.figure("2D Data with SVM")
        plt.xlabel("First Feature X1",fontsize=18)
        plt.ylabel("Second Feature X2",fontsize=18)
        plt.plot(X1_train[:,0], X1_train[:,1], "ro")
        plt.plot(X2_train[:,0], X2_train[:,1], "bo")
        plt.scatter(model.sv[:,0], model.sv[:,1], s=100, c="g",marker="o")
    
       
        plt.axis("tight")
        
        # w.x + b = 0
        a0 = -2; a1 = f(a0, model.w, model.b)
        b0 = 5; b1 = f(b0, model.w, model.b)
        plt.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -2; a1 = f(a0, model.w, model.b, 1)
        b0 = 5; b1 = f(b0, model.w, model.b, 1)
        plt.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -2; a1 = f(a0, model.w, model.b, -1)
        b0 = 5; b1 = f(b0, model.w, model.b, -1)
        plt.plot([a0,b0], [a1,b1], "k--")

      
        plt.show()
     
    def plot_data(X,y):                    # this function we added for plotting the 2D dataset
        
        plt.figure("2D Data")
        plt.xlabel("First Feature X1",fontsize=18)
        plt.ylabel("Second Feature X2",fontsize=18)
        plt.scatter(X[:,0], X[:,1], s=50, c=y, marker="o")
        plt.show()    
        
    def plot_contour(X1_train, X2_train, model):
        
        plt.figure("SVM With Kernal")
        plt.xlabel("First Feature X1",fontsize=18)
        plt.ylabel("Second Feature X2",fontsize=18)
        plt.plot(X1_train[:,0], X1_train[:,1], "ro")
        plt.plot(X2_train[:,0], X2_train[:,1], "bo")
        plt.scatter(model.sv[:,0], model.sv[:,1], s=100, c="g",marker="o")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z,A=model.predict(X)
        Z = Z.reshape(X1.shape)
        plt.contour(X1, X2, Z, [0.0], colors='b', linewidths=1, origin='lower')
        plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        plt.axis("tight")
        plt.show()
        
        
        

    def linear_svm():
        X1, y1, X2, y2 = lin_separable_data()
        X_train, y_train, X_test, y_test = split_train_test(X1,y1,X2,y2)
        
        
        plot_data(X_train,y_train)
        
        model = SVM(linear_kernel)                  
        model.fit(X_train, y_train)
        
        y_predict_value,y_predict = model.predict(X_train)
        correct = np.sum(y_predict == y_train)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("Accuracy on Training Data=", correct/len(y_predict))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], model)
        
        y_predict_value,y_predict = model.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("Accuracy on Test Data=", correct/len(y_predict))
        
        
    def kernel_svm():
        X1, y1, X2, y2 = circular_data()
        #X1, y1, X2, y2 = lin_separable_data()
        #X1, y1, X2, y2 = lin_separable_overlap_data()
        X_train, y_train, X_test, y_test = split_train_test(X1,y1,X2,y2)
        
        plot_data(X_train,y_train)

        model = SVM(polynomial_kernel)              #Here you can switch the kernel function
        model.fit(X_train, y_train)

        y_predict_value,y_predict = model.predict(X_train)
        correct = np.sum(y_predict == y_train)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("Accuracy on Training Data=", correct/len(y_predict))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], model)
        
        y_predict_value,y_predict = model.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("Accuracy on Test Data=", correct/len(y_predict))
        
    def soft_svm():
        X1, y1, X2, y2 = lin_separable_overlap_data()
        X_train, y_train, X_test, y_test = split_train_test(X1,y1,X2,y2)
        
        
        plot_data(X_train,y_train) 
        
        
        model = SVM(polynomial_kernel,C=500.2)         #Here you can switch the kernel function
        model.fit(X_train, y_train)
        
        y_predict_value,y_predict = model.predict(X_train)
        correct = np.sum(y_predict == y_train)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("Accuracy on Training Data=", correct/len(y_predict))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], model)
        
        y_predict_value,y_predict = model.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("Accuracy on Test Data=", correct/len(y_predict))
        
        
        
        
    # after you have implemented the kernel and fit functions let us test the implementations
    # uncomment each of the following lines as and when you have completed their implementations.
    
    #please enter the choice from the console/terminal
    
    print("Enter the choice 1. linear SVM, 2. Kernel SVM, 3. Soft SVM")
    choice=int(input())
    if(choice==1):
        linear_svm()
    elif(choice==2):    
        kernel_svm()
    elif(choice==3):    
        soft_svm()
    else:
        print("Wrong Choice")
        
#    linear_svm()
#    kernel_svm()
#    soft_svm()    


