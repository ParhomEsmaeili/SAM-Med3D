
#Here we place a sample implementation for the basic requirements and the provided arguments for an inference app. We only provide the dataset name and the inference
#device. Latter is provided such that any built models can be placed onto the inference device.

import torch 
from monai.data import MetaTensor 

class InferApp:
    def __init__(self,
                dataset_name:str,
                infer_device:torch.device
                ):
        
        self.dataset_name = dataset_name
        self.infer_device  = infer_device 

        self.build_inference_apps()

    def heuristic_planning_build(self):
        #Exempler optional method: If you wanted to put any functionality for extracting your build config dependent on task etc. 
        pass 

    def dummy_inference(self, request):
        '''
        
            NOTE: Checks will be put in place to ensure that image resolution, spacing, orientation will be matching & otherwise 
        the code will be non-functional.

            'logits': Dict which contains the following fields:

                'metatensor': MetaTensor or torch object, ((torch.float dtype)), multi-channel logits map (CHWD), where C = Number of Classes (channel first format)
            
                'meta_dict: Meta information in dict format,  ('affine must match the input-images' affine info)
            
            'pred': Dict which contains the following fields:
                metatensor: MetaTensor or torch tensor object ((torch.int dtype)) containing the discretised prediction (shape 1HWD)
                meta_dict: Meta information in dict format, which corresponds to the header of the prediction (affine array must match the input image's meta-info)

            NOTE: The meta dictionaries will be expected to contain a key:item pair denoted as "affine", containing the 
            affine array required for saving the segmentations in ITK format.
            
            NOTE: The affine must be a torch tensor or numpy array.

        NOTE: These outputs must be stored/provided on cpu. 
        '''
        #Returns an output with the same structure as the input image tensor. 
        
        #dummy logits just use a set of slices at different offsets.. 
        
        img = request['image']['metatensor']
        shape = img.shape 

        list_logits = []
        for idx, class_lb in enumerate(request['config_labels_dict'].keys()):
            #create dummy
            # dummy = torch.zeros_like(img)
            dummy = torch.from_numpy(34.1103 * (idx + 1) * request['image']['metatensor'].array).to(dtype=torch.float64) 
            list_logits.append(dummy)
        
        logits_tensor = torch.cat(list_logits, dim=0).to(dtype=torch.float64)

        #Pred = Plain centred binary mask, like a cuboid.
        
        pred = torch.zeros(img.shape)
        pred[0, int(shape[1]/2 - 20) : int(shape[1]/2 + 5),  int(shape[2]/2 - 20) : int(shape[2]/2 + 5), int(shape[3]/2 - 20): int(shape[3]/2 + 5)] = 1
        
        pred.to(dtype=torch.int64) 


        output = {
            'logits':{
                'metatensor':logits_tensor,
                'meta_dict':{'affine': request['image']['meta_dict']['affine']}
            },
            'pred':{
                'metatensor':pred,
                'meta_dict':{'affine': request['image']['meta_dict']['affine']}
            },
        }

        
        #Create the output dict. 
        return output 
    
    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':self.dummy_inference,
            'IS_interactive_init':self.dummy_inference,
            'IS_interactive_edit':self.dummy_inference
            }
    def app_configs(self):

        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return {'infer_app_name': 'Sample_TEST'} 
    
    def __call__(self, request: dict):
        
        '''
        Input request dictionary for application contains the following input fields:

        NOTE: All input arrays, tensors etc, will be on CPU. NOT GPU. 

        NOTE: Orientation convention is always assumed to be RAS! 

            image: A dictionary containing a path & a pre-loaded (UI) metatensor objects 
            {'path':image_path, 
            'metatensor':monai metatensor object containing image, torch.float datatype.
            'meta_dict': image_meta_dictionary, contains the original affine from loading and current affine array in the ui-domain.}

            model: A string denoting the inference "mode" being simulated, has three options: 
                    1) Automatic Segmentation, denoted: 'IS_autoseg' 
                    2) Interactive Initialisation: 'IS_interactive_init'
                    3) Interactive Editing: 'IS_interactive_edit'
            
            config_labels_dict: A dictionary containing the class label - class integer code mapping relationship being used.

            im: An interaction memory dictionary containing the set of interaction states. 
            Keys = Infer mode + optionally (for Edit) the inference iter num (1, ...).       

            Within each interaction state in IM:    
            
            prev_logits: A dictionary containing: {
                    'paths': list of paths, to each individual logits map (HWD), in the same order as provided by output CHWD logits map}
                    'metatensor': Non-modified (CHWD) metatensor/torch tensor that is forward-propagated from the prior output (CHWD).
                    'meta_dict': Non-modified meta dictionary that is forward propagated.
                    }
            prev_pred: A dictionary containing: {
                    'path': path to the discretised map (HWD)}
                    'metatensor': Non-modified metatensor/torch tensor that is forward-propagated from the prior output (1HWD).
                    'meta_dict': Non-modified meta dictionary that is forward propagated.
                    }

            prompt information: See `<https://github.com/IS_Validate/blob/main/src/data/interaction_state_construct.py>`

            interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor] format 
                {'interactions':dict[prompt_type_str[list[torch.tensor] OR NONE ]], 
                'interactions_labels':dict[prompt_type_str_labels [list[torch.tensor] OR NONE]],
                }
            interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
                (where each prompt spatial coord is represented as a sublist).  
                dict[prompt_type_str[class[list[list]] OR NONE]]

        '''
        #Callback which is required, and will be what is used to process the input requests for inference. E.g.:
        
        app = self.infer_apps[request['model']]
        return app(request)


if __name__ == '__main__':
    infer_app = InferApp('BraTS2021', torch.device('cpu'))

    infer_app.app_configs()

    request = {
        'image':{
            'path':'dummy',
            'metatensor': MetaTensor(torch.ones((1,100,100,100))),
            'meta_dict':{'affine':torch.eye(4)}
        },
        'model':'IS_interactive_edit',
        'config_labels_dict':{'tumour':1, 'background':0},
        'im':{'Automatic Init':None},

    }
    infer_app(request)
    print('halt')