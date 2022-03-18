def plot_tree(dtree, feature_names=None):
    '''
    Visualize a sklearn's DecisionTree `dtree` as printed text.
    `feature_names` accepts a `list` of `str`.
    '''
    
    dtree = dtree.tree_
    def rec(ii,dep=0):
        feature = dtree.feature[ii]
        if feature>-1 and feature_names is not None:
            feature = feature_names[feature]
        print(
            '|'*dep, dep, ' ', 
            # dtree.value[ii]*1000//dtree.n_node_samples[ii]/10, '\t',
            dtree.value[ii]*1000//dtree.value[0]/10, '\t',
            int(dtree.impurity[ii]*1000)/10, '\t',
            feature, ' <= ', dtree.threshold[ii],
            sep='',
        )
        if dtree.children_left[ii]>0:
            rec(dtree.children_left[ii], dep+1)
        if dtree.children_right[ii]>0:
            rec(dtree.children_right[ii], dep+1)
    rec(0)
