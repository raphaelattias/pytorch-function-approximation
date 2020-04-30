addpath('/Users/raphael-attias/Downloads/npy-matlab-master')
savepath
%bf = readNPY('bf.npy');
[X,Y,Z] = meshgrid(0:35,0:35,0:35);
xslice = [0,18];
yslice = 0;
zslice = 0:18:35;
slice(X,Y,Z,bf,xslice,yslice,zslice);
alpha 1
hold off

