clc;close();



[xx,yy]=ndgrid(-10:10,-10:10);
z = yy;

figure
surf(xx,yy,z)
hold on;

[xx,yy]=ndgrid(-10:10,-10:10);
z = xx;
surf(xx,yy,z)

[xx,z]=ndgrid(-10:10,-10:10);
yy = xx;
surf(xx,yy,z)