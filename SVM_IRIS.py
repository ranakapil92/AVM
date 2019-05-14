# check the following link for using cvxopt qp solver
# http://cvxopt.org/examples/tutorial/qp.html


import numpy as np
from numpy import linalg
from csv import reader                                      #for reading the fiile    
import cvxopt
import cvxopt.solvers
from pandas import DataFrame                                #for data arrangement
import matplotlib.pyplot as plt
import math
        
get_ipython().run_line_magic('matplotlib', 'qt')

def linear_kernel(x1, x2):   
        temp=np.dot(x1,x2)
        return temp

# the reason of adding it , because linear kernal is passed in SVM  class initialization explicitly. and class is 
# was compling  before linear_kernal          
    
class SVM(object):

    def __init__(self, kernel, C=None):
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
            return temp, np.sign(temp)
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
    
    
    from sklearn.model_selection import train_test_split 
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        # to do
        
    def linear_kernel(x1, x2):
        temp=np.dot(x1,x2)
        return temp
        
    
    def polynomial_kernel(x1, x2, q=2):
        # implement the polynomial kernel
        # to do
        temp = (1 + np.dot(np.transpose(x1), x2)) ** q
        return temp
    
    def gaussian_kernel(x1, x2, s=2):
        # implement the radial basis function kernel
        # to do
        temp = np.exp(-linalg.norm(x1-x2)**2 / (2 * (s ** 2)))
        return temp
    
    def get_IRIS_data():
        inputdata = list()
        with open("iris.data", 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                inputdata.append(row)        
        
        dataset = DataFrame(inputdata)
        
        return dataset
    
    def two_class_data(Xtrain,ytrain,Xtest,ytest,c):
        
        for i in range(0,len(Xtrain)):
            if(ytrain[i]==c):
                ytrain[i]=1
            else:
                ytrain[i]=-1
        for i in range(0,len(Xtest)):
            if(ytest[i]==c):
                ytest[i]=1
            else:
                ytest[i]=-1        
        return Xtrain,ytrain,Xtest,ytest
        
    
            
        
    def plot_contour(X,y,m1,m2,m3):
        
        plt.figure("IRIS Data With First 2 Features")
        plt.xlabel("First Feature X1",fontsize=18)
        plt.ylabel("Second Feature X2",fontsize=18)
        for i in range(0,len(X)):
            if(y[i]==1):
                plt.plot(X[i,0], X[i,1], "ro")
            elif(y[i]==2):
                plt.plot(X[i,0], X[i,1], "bo")
            elif(y[i]==3):
                plt.plot(X[i,0], X[i,1], "yo")
        plt.axis("tight")
        
        X3, X4 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        Xz = np.array([[x1, x2, 0, 0] for x1, x2 in zip(np.ravel(X3), np.ravel(X4))])
        Z,A=m1.predict(Xz)
        Z = Z.reshape(X3.shape)
        plt.contour(X3, X4, Z, [0.0], colors='b', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
        
        Z,A=m2.predict(Xz)
        Z = Z.reshape(X3.shape)
        plt.contour(X3, X4, Z, [0.0], colors='r', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')


        Z,A=m3.predict(Xz)
        Z = Z.reshape(X3.shape)
        plt.contour(X3, X4, Z, [0.0], colors='g', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')


        plt.axis("tight")
        plt.show()
        
        
        plt.show()
        
        plt.figure("IRIS Data With Last 2 Features")
        plt.xlabel("Third Feature X3",fontsize=18)
        plt.ylabel("Fourth Feature X4",fontsize=18)
        for i in range(0,len(X)):
            if(y[i]==1):
                plt.plot(X[i,2], X[i,3], "ro")
            elif(y[i]==2):
                plt.plot(X[i,2], X[i,3], "bo")
            elif(y[i]==3):
                plt.plot(X[i,2], X[i,3], "yo")
        plt.axis("tight")
        
        X3, X4 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        Xz = np.array([[x1, x2, 0, 0] for x1, x2 in zip(np.ravel(X3), np.ravel(X4))])
        Z,A=m1.predict(Xz)
        Z = Z.reshape(X3.shape)
        plt.contour(X3, X4, Z, [0.0], colors='b', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        
        Z,A=m2.predict(Xz)
        Z = Z.reshape(X3.shape)
        plt.contour(X3, X4, Z, [0.0], colors='r', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')


        Z,A=m3.predict(Xz)
        Z = Z.reshape(X3.shape)
        plt.contour(X3, X4, Z, [0.0], colors='g', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(X3, X4, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
        
        
        plt.axis("tight")
        plt.show()
        
        

        
    def soft_svm():
        dataset = get_IRIS_data()
        for i in range(0,len(dataset)):
            if(dataset.iloc[i][4]=="Iris-setosa"):    # 1 = Iis-setosa
                dataset.iloc[i][4]=1
            elif(dataset.iloc[i][4]=="Iris-versicolor"):     # 2 = Iris-versicolor
                dataset.iloc[i][4]=2    
            elif(dataset.iloc[i][4]=="Iris-virginica"):      # 3 = Iris-virginica
                dataset.iloc[i][4]=3      
        for i in range(0,5):
            dataset[i]= dataset[i].astype(float) 
           
        
        X_main = dataset.iloc[:,:4].values
        y_main = dataset.iloc[:,4:5].values
        X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.25)
        
        #instead of writting it iteratively for each class , I am writing in very simple way for because we have only 3 casses
        #so test_class here defining the class for which y_label is equal to +1 and else are -1.
        
        
        
        k=gaussian_kernel
        
        #---------------SVM for class=1--------------------------------------------------------------
        
        test_class=1
        X_train1 = np.copy(X_train)
        y_train1=np.copy(y_train)
        X_test1=np.copy(X_test)
        y_test1=np.copy(y_test)
        
        
        
        X_train1,y_train1,X_test1,y_test1 = two_class_data(X_train1,y_train1,X_test1,y_test1,test_class)
    
        model1 = SVM(k,C=1000.1)
        model1.fit(X_train1, y_train1)
        
        y_predict_value1,y_predict1 = model1.predict(X_test1)
        
        
        
        
        
        #---------------SVM for class=2--------------------------------------------------------------
        test_class=2
        X_train1 = np.copy(X_train)
        y_train1=np.copy(y_train)
        X_test1=np.copy(X_test)
        y_test1=np.copy(y_test)
        
        X_train1,y_train1,X_test1,y_test1 = two_class_data(X_train1,y_train1,X_test1,y_test1,test_class)
        

        model2 = SVM(k,C=1000.1)
        model2.fit(X_train1, y_train1)

        y_predict_value2, y_predict2 = model2.predict(X_test1)
        
        
        #---------------SVM for class=3--------------------------------------------------------------
        
        test_class=3
        X_train1 = np.copy(X_train)
        y_train1=np.copy(y_train)
        X_test1=np.copy(X_test)
        y_test1=np.copy(y_test)
        
        X_train1,y_train1,X_test1,y_test1 = two_class_data(X_train1,y_train1,X_test1,y_test1,test_class)
        

        model3 = SVM(k,C=1000.1)
        model3.fit(X_train1, y_train1)

        y_predict_value3,y_predict3 = model3.predict(X_test1)
        
        plot_contour(X_train,y_train, model1,model2,model3)
        
        correct=0
        y_predicted_class=np.zeros(len(y_test))     
        
        for i in range(0,len(y_test)):
            if((y_predict3[i]>y_predict2[i]) and (y_predict3[i] >y_predict1[i])):
                y_predicted_class[i]=3
                if(y_test[i]==3):
                    correct=correct + 1
            elif(y_predict2[i]>y_predict1[i]):
                y_predicted_class[i]=2
                if(y_test[i]==2):
                    correct=correct + 1
            else:
                y_predicted_class[i]=1
                if(y_test[i]==1):
                    correct=correct + 1
                    
         
        
        
        for i in range(0,len(y_test)):
            print("predicted Class-Actaual Class==%d,%d" % (y_predicted_class[i],y_test[i]))
       
        print("total correct=%d out of %d" % (correct,len(y_test)))   
        print("Accuracy=",correct/len(y_test))
        
       
   
    soft_svm()    


