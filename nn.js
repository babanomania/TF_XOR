class NNModel {

    constructor(){
        this.tfmodel = tf.sequential();

        this.tfmodel.add(  tf.layers.dense({
            units: 2, inputShape: [2], activation: 'sigmoid'
        }) );

        this.tfmodel.add(  tf.layers.dense({
            units: 1, activation: 'sigmoid'
        }) );

        const LEARNING_RATE = 0.5;
        const optimizer = tf.train.sgd(LEARNING_RATE);

        this.tfmodel.compile({
            loss: 'meanSquaredError', optimizer: optimizer
        });
    }

    async train(){

        const input_x = [
            [ 0, 0 ],
            [ 0, 1 ],
            [ 1, 0 ],
            [ 1, 1 ],
        ];

        const input_y = [
            [ 0 ],
            [ 1 ],
            [ 1 ],
            [ 0 ],
        ];

        var tns_ip_x = tf.tensor2d( input_x);
        var tns_ip_y = tf.tensor2d( input_y);

        await this.tfmodel.fit( 
            tns_ip_x,
            tns_ip_y,
            { shuffle: true, },
        );

        //console.log( "loss ", fit_resp.history.loss[99] );
        tns_ip_x.dispose();
        tns_ip_y.dispose();
        
    }

    async predict( rid, cid ){

        var tns_ip = tf.tensor([ 
            [ rid, cid ]
        ]);

        var tns_op = await this.tfmodel.predict( tns_ip )
        var prd_op = await tns_op.data();

        tns_ip.dispose();
        tns_op.dispose();

        //console.log( "rid ", rid * cell_width, ", cid ", cid * cell_width, ", color ", Math.floor( prd_op[0] * 255 ) );
        return prd_op[0];
    }
}