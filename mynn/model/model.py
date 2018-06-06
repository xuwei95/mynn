import pickle
class model(object):
    def __init__(self):
        self.layerlist=[]
        self.layernum=len(self.layerlist)
    def add(self,layer):
        self.layerlist.append(layer)
    def predict(self,x):
        out=x
        for layer in self.layerlist[:-1]:
            out=layer.forward(out)
        return out
    def fit(self,x,y,alpha=0.00001):
        out=x
        for layer in self.layerlist[:-1]:
            out=layer.forward(out)
        cost=self.layerlist[-1]
        loss=cost.cal_loss(out,y)
        eta=cost.gradient()
        self.layernum=len(self.layerlist)
        a=self.layernum-2
        while a>=0:
            eta=self.layerlist[a].gradient(eta)
            a-=1
        for layer in self.layerlist:
            if layer.layertype == 'fc' or layer.layertype == 'conv':
                layer.backward(alpha=alpha)
        return loss
    def save(self,name):
        file = open(name, 'wb')
        pickle.dump(self.layerlist,file)
        file.close()
    def load(self,name):
        file = open(name, 'rb')
        self.layerlist=pickle.load(file)
        file.close()
