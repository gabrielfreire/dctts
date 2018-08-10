import tensorflowjs as tfjs
import tensorflow as tf

def save (model_dir, export_dir, is_saved_model=False, scope='Text2Mel'):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if (not is_saved_model):    
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
        builder.save()

def convert (model_dir, output_node_names='Sigmoid', output_dir):
    tfjs.converters.convert_tf_saved_model(model_dir, output_node_names=output_node_names, output_dir=output_dir, saved_model_tags='serve', skip_op_check=True, strip_debug_ops=True)