class NNModel {

    constructor(){
        this.tfmodel = tf.sequential();

        this.tfmodel.add(  tf.layers.dense({
            units: 2, inputShape: [2]
        }) );

        this.tfmodel.add(  tf.layers.dense({
            units: 1
        }) );

        this.tfmodel.compile({
            loss: 'meanSquaredError', optimizer: 'sgd'
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

        var fit_resp = await this.tfmodel.fit( 
            tf.tensor2d( input_x),
            tf.tensor2d( input_y ),
            { shuffle: true, },
        );

        //console.log( "loss ", fit_resp.history.loss[99] );
    }

    async predict( rid, cid ){

        var prd_op = await this.tfmodel.predict(
            tf.tensor([ 
                [ rid, cid ]
            ]),

        ).data();

        //console.log( "rid ", rid * cell_width, ", cid ", cid * cell_width, ", color ", Math.floor( prd_op[0] * 255 ) );
        return prd_op[0];
    }
}