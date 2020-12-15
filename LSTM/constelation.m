function constelation()

global sys;
global var;
    for a = 0:1:2^(sys.data_rate)-1
        obj=dec2bin(a,sys.data_rate);
        c=[];
            for i=1:length(obj)
                c=[c str2num(obj(i))];
            end
        var.sample(:,a+1) = c';
        var.samples(:,a+1) = modulator(c',sys.data_rate);
    end
    
    var.id = eye(2^sys.data_rate);
end