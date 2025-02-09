classdef multilabelClassificationLayer < nnet.layer.RegressionLayer
    properties
        ClassWeights
    end

    methods
        function layer = multilabelClassificationLayer(name, classWeights)
            layer.Name = name;
            layer.ClassWeights = classWeights;
            layer.Description = "Binary cross-entropy loss layer for multi-label classification";
        end

        function loss = forwardLoss(~, Y, T)
            % Binary cross-entropy loss for multi-label classification
            eps = 1e-8; % Prevent log(0)
            %layer.ClassWeights .* 
            loss = -sum((T .* log(Y + eps) + (1 - T) .* log(1 - Y + eps)), 'all') / numel(T);
        end
    end
end
