module myFun

    export MRIscan

    @everywhere function MRIscan(T2valueForM1::Float64,T2valueForM2::Float64,T2valueForM3::Float64,FrequencyOffsetForM2::Float64,FrequencyOffsetForM3::Float64,FOV::Float64,DW::Float64,matrixSizeX::Int64,matrixSizeY::Int64,TE::Float64,NoiseLevel::Float64)
        B0offsetForM2 = FrequencyOffsetForM2
        B0offsetForM3 = FrequencyOffsetForM3
        matrixSize = [matrixSizeX,matrixSizeY]
        TE = max(TE,0.01)
        DW = 50e-6; # hard-coded for now
        #DW = min(DW, 100e-6)
        #DW = max(DW, 5e-6)
        matrixSizeX = max(matrixSizeX,32)
        matrixSizeX = min(matrixSizeX,128)
        matrixSizeY = max(matrixSizeX,32)
        matrixSizeY = min(matrixSizeX,128)
        fid = open("phantoms/m1.img"); m1 = read(fid,UInt8,128*128); close(fid)
        m1 = convert(Array{Float64},reshape(m1,128,128));
        fid = open("phantoms/m2.img"); m2 = read(fid,UInt8,128*128); close(fid)
        m2 = convert(Array{Float64},reshape(m2,128,128));
        fid = open("phantoms/m3.img"); m3 = read(fid,UInt8,128*128); close(fid)
        m3 = convert(Array{Float64},reshape(m3,128,128));
        ProtonDensityMap = m1+m2+m3;
        T2map = zeros(128,128);T2map[find(m1.>=2)]=T2valueForM1; T2map[find(m2.>=2)]=T2valueForM2; T2map[find(m3.>=3)]=T2valueForM3; 
        B0offsetmap = zeros(128,128); B0offsetmap[find(m2.>=2)]=B0offsetForM2; B0offsetmap[find(m3.>=3)]=B0offsetForM3; 
        PHANTOMSIZE = 240; # mm
        xx = size(ProtonDensityMap)[1]; yy = size(ProtonDensityMap)[2];
        vx = PHANTOMSIZE/xx; vy = PHANTOMSIZE/yy;
        locationX = repmat(-vx*xx/2:vx:vx*xx/2-1,1,yy);
        locationY = repmat((-vy*yy/2:vy:vy*yy/2-1)',xx,1);

        d1 = 5e-3 # pre-phasing gradient duration
        BW = 1/DW
        ReadoutAcquisitionWindow = DW*matrixSize[1]
        d2 = TE - (ReadoutAcquisitionWindow/2.)-d1
        GradientX = BW/FOV   # Hz/mm
        GradientYi = (GradientX*DW)/d1
        GradientX1 = -GradientYi*(round(matrixSize[1]/2))
        adcStartTime = d1+d2+DW
        adcEndTime = d1+d2+ReadoutAcquisitionWindow;

        @everywhere function Kscan(t,kx,ky,pd,T2,B0,X,Y)
            kdata = zeros(Complex{Float64},size(t))
            i = complex(0,1)
            for cnt = 1:size(t)[1]
                kdata[cnt]=sum(pd.*exp.(-t[cnt]./T2).*exp.(i*2*π*B0*t[cnt]).*exp.(i*2*π*kx[cnt].*X).*exp.(i*2*π*ky[cnt].*Y))
            end
            return kdata
        end

    # 	kSpaceData2D = zeros(Complex{Float64,},matrixSize[1],matrixSize[2])
        kSpaceData2D = SharedArray{Complex{Float64,}}((matrixSize[1],matrixSize[2]))

        timeSampled = 0:DW:adcEndTime
        acquisition_timeSampled = timeSampled[1+size(DW:DW:d1)[1]+size(d1+DW:DW:d1+d2)[1]+1:end];

        GxWaveform = vcat(0.,GradientX1*ones(size(DW:DW:d1)),zeros(size(d1+DW:DW:d1+d2)), GradientX*ones(size(adcStartTime:DW:adcEndTime)));
        trajectory_kx = cumsum(GxWaveform)*DW;
        acquisition_trajectory_kx = trajectory_kx[1+size(DW:DW:d1)[1]+size(d1+DW:DW:d1+d2)[1]+1:end];

        @sync @parallel  for TRcount = 1:matrixSize[2]
            kyLocation = TRcount-round(matrixSize[2]/2)
            GyWaveform = vcat(0.,GradientYi*kyLocation.*ones(size(DW:DW:d1)),zeros(size(d1+DW:DW:d1+d2)), zeros(size(adcStartTime:DW:adcEndTime)));
            trajectory_ky = cumsum(GyWaveform)*DW;
            acquisition_trajectory_ky = trajectory_ky[1+size(DW:DW:d1)[1]+size(d1+DW:DW:d1+d2)[1]+1:end];
            kdata = Kscan(acquisition_timeSampled,acquisition_trajectory_kx,acquisition_trajectory_ky,ProtonDensityMap[:],T2map[:],B0offsetmap[:],locationX[:],locationY[:]);
            kSpaceData2D[:,TRcount] = kdata;
        end
    kSpaceData2D = kSpaceData2D + NoiseLevel*randn(size(kSpaceData2D)) + NoiseLevel*complex(0,1)*randn(size(kSpaceData2D))
        return kSpaceData2D
    end

end
