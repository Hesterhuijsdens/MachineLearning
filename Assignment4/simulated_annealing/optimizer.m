% problem definition
% minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
% w is a symmetric real n x n matrix with zero diagonal

METHOD='sa';
NEIGHBORHOODSIZE=1;
n_restart =10;


switch METHOD,
case 'iter'

	E_min = 1000;
	for t=1:n_restart,

		% initialize
		x = 2*(rand(1,n)>0.5)-1;
		E1 = E(x,w);
		flag = 1;
	
		while flag == 1,
			flag = 0;
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one bit i
				% compute dE directly instead of subtracting E's of
				% different states because of efficiency
			case 2,
				% choose new x by flipping bits i,j
			end;
		end;
        E1
        figure(1)
        hold on
        for i=1:n
            if x(i)==1
                plot(z(i,1),z(i,2),'r*')
            else
                plot(z(i,1),z(i,2),'b*')
            end;
        end;
        hold off
        pause
		E_min = min(E_min,E1);
	end;
	E_min
case 'sa'
	% initialize
	x = 2*(rand(1,n)>0.5)-1;
	E1 = E(x,w);
	E_outer=zeros(1,100);	%stores mean energy at each temperature
	E_bar=zeros(1,100);		% stores std energy at each temperature

	% initialize temperature
	max_dE=0;
	switch NEIGHBORHOODSIZE,
        case 1,
			% estimate maximum dE in single spin flip
        case 2,
			% estimate maximum dE in pair spin flip
        end;
	beta_init=1/max_dE;	% sets initial temperature
	T1=1000; % length markov chain at fixed temperature
	factor=1.05 ; % increment of beta at each new chain

	beta=beta_init;
    disp(beta)
	E_bar(1)=1;
	t2=1;
	while E_bar(t2) > 0,
		t2=t2+1;
		beta=beta*factor;
		E_all=zeros(1,T1);
		for t1=1:T1,
			switch NEIGHBORHOODSIZE,
			case 1,
				% choose new x by flipping one random bit i
				% perform Metropolis Hasting step
			case 2,
				% choose new x by flipping random bits i,j
				% perform Metropolis Hasting step
			end;
			% E1 is energy of new state
			E_all(t1)=E1;
		end;
        disp(beta)
		E_outer(t2)=mean(E_all);
		E_bar(t2)=std(E_all);
		%[t2 beta E_outer(t2) E_bar(t2)] % observe convergence
	end;
    disp(beta)
	%E_min=E_all(1) % minimal energy 
    %plot(1:t2,E_outer(1:t2),1:t2,E_bar(1:t2))
end;

