rand('state',3)

n=100;
p=1.0/n;
w=sprandsym(n, p);
%w=(w>0)-(w<0); % this choice defines a frustrated system
%disp(w)
w=(w>0); % this choice defines a ferro-magnetic (easy) system
%nnz(w)
w=w-diag(diag(w));
%disp(w)
%nnz(w)


