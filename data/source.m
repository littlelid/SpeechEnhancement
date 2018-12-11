cut and paste from "openExample('phased/GeneralizedSidelobeCancellationOnULAExample')" in MATLAB

c = 340.0;
fs = 8.0e3;
fc = fs/2;
lam = c/fc;
transducer = phased.OmnidirectionalMicrophoneElement('FrequencyRange',[20 20000]);
array = phased.ULA('Element',transducer,'NumElements',11,'ElementSpacing',lam/2);

t = 0:1/fs:.5;
signal = chirp(t,0,0.5,500);

collector = phased.WidebandCollector('Sensor',array,'PropagationSpeed',c, 'SampleRate',fs,'ModulatedInput',false,'NumSubbands',512);
incidentAngle = [-50;0];
signal = collector(signal.',incidentAngle);
noise = 0.5*randn(size(signal));
recsignal = signal + noise;
