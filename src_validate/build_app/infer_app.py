
#Here we place a sample implementation for the basic requirements and the provided arguments for an inference app. We only provide the dataset name and the inference
#device. Latter is provided such that any built models can be placed onto the inference device.

import torch 
from monai.data import MetaTensor 
import os
import sys 
basedir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(basedir)
import copy 

from segment_anything import sam_model_registry
from segment_anything.build_sam3D import sam_model_registry3D
# from segment_anything.utils.transforms3D import ResizeLongestSide3D
import torch
import torch.nn.functional as F
import torchio as tio
import re 

class InferApp:
    def __init__(self,
                dataset_info:dict,
                infer_device:torch.device
                ):
        
        self.dataset_info = dataset_info
        self.infer_device  = infer_device 

        self.app_params = {
            'model_type':'vit_b_ori',
            'checkpoint_name': 'sam_med3d_turbo.pth'
        }
        self.target_spacing = (1.5, 1.5, 1.5)
        self.patch_size = (128, 128, 128)


        self.load_model()
        self.build_inference_apps()
    
    def load_model(self):
        self.sam_model = sam_model_registry3D[self.app_params['model_type']](checkpoint=None).to(
            self.infer_device
        )
        if self.app_params['checkpoint_name'] is not None:
            ckpt_path = os.path.join(basedir, 'ckpt', self.app_params['checkpoint_name'])
            model_dict = torch.load(ckpt_path, map_location=self.infer_device, weights_only=False)
            state_dict = model_dict["model_state_dict"]
            self.sam_model.load_state_dict(state_dict)
        else:
            raise Exception 
    
    def binary_pad_inference(self):
        # subject = tio.Subject(
        # image=tio.ScalarImage(tensor=copy.deepcopy(request['image']['metatensor'].array)),
        # )
        # crop_transform = tio.CropOrPad(
        #                             target_shape=(128, 128, 128))
        # padding_params, cropping_params = crop_transform.compute_crop_or_pad(
        #     subject)
        # if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
        # if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)


        # infer_transform = tio.Compose([
        #     crop_transform,
        #     tio.ZNormalization(masking_method=lambda x: x > 0),
        # ])
        # subject_roi = infer_transform(subject)

        # img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        #     1), subject_roi.label.data.clone().detach().unsqueeze(1)
        # ori_roi_offset = (
        #     cropping_params[0],
        #     cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        #     cropping_params[2],
        #     cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        #     cropping_params[4],
        #     cropping_params[4] + 128 - padding_params[4] - padding_params[5],
        # )

        # meta_info = {
        #     "image_path": img_path,
        #     "image_shape": sitk_image_arr.shape[1:],
        #     "origin": sitk_label.GetOrigin(),
        #     "direction": sitk_label.GetDirection(),
        #     "spacing": sitk_label.GetSpacing(),
        #     "padding_params": padding_params,
        #     "cropping_params": cropping_params,
        #     "ori_roi": ori_roi_offset,
        # }
        # return (
        #     img3D_roi,
        #     gt3D_roi,
        #     meta_info,
        # )
        raise NotImplementedError



    def binary_sw_inference(self, subject):
        '''
        
            NOTE: Checks will be put in place to ensure that image resolution, spacing, orientation will be matching & otherwise 
        the code will be non-functional.

            'probs': Dict which contains the following fields:

                'metatensor': MetaTensor or torch object, ((torch.float dtype)), multi-channel probs map (CHWD), where C = Number of Classes (channel first format)
            
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
        
        #Here we will perform the grid sampling.

        #We extract some basic parameters for the overlap according to the size of the resampled image: We select 0 as the lower bound so that if it doesn't need to
        #overlay then it won't, this is absolutely not a good heuristic but we will use this for now for testing our framework. For instances where dim < 128 we pick the
        #smallest even value s.t. dim + pad >= 128.

        pad_l = lambda x,y: 0 if abs(y) >= abs(x) else (x - y) % 2 + (x-y)  

        pad_dim1 = pad_l(self.patch_size[0], subject['image'].shape[1], )
        pad_dim2 = pad_l(self.patch_size[1], subject['image'].shape[2])
        pad_dim3 = pad_l(self.patch_size[2], subject['image'].shape[3])

        #We disentangle the image and prompt components of the subject into distinct subject obj so that we can pass them through separate grid samplers.
        im_subj = tio.Subject(image=subject['image'])
        prompt_subj = tio.Subject(label=subject['label'])

        #Implementing an image-level intensity normalisation in-line with the approach that the SAM-Med3D authors take.
        intensity_transforms = tio.Compose([tio.Clamp(-1000,1000), tio.ZNormalization(masking_method=lambda x: x > 0)]) if self.dataset_info['dataset_modality'].title() == 'CT' else tio.Compose([tio.ZNormalization(masking_method=lambda x: x > 0)])
        im_subj = intensity_transforms(im_subj)


        #We create separate grid samplers for the image and the prompts, same padding dimensions but different padding modes since prompts should have no injected info
        #meanwhile it is probably best to use transform which will minimise the shift in the histogram of the image patch.

        im_grid_sampler = tio.inference.GridSampler(im_subj, self.patch_size, (pad_dim1, pad_dim2, pad_dim3), 0)
        prompt_grid_sampler = tio.inference.GridSampler(prompt_subj, self.patch_size, (pad_dim1, pad_dim2, pad_dim3), 0)

        #We elect to just pad with zero and then crop the overlapping regions anyways. This is probably not ideal, nevertheless it will likely ensure 
        # that the distribution of the voxel intensities will be shifted in a similar capacity regardless of the voxel intensities of the border.

        im_patch_loader = tio.SubjectsLoader(im_grid_sampler, batch_size=1)
        prompt_patch_loader = tio.SubjectsLoader(prompt_grid_sampler, batch_size=1)

        aggregator = tio.inference.GridAggregator(im_grid_sampler, overlap_mode='crop')

        for i, (patch1, patch2) in enumerate(zip(im_patch_loader, prompt_patch_loader)):
            # print(patch1)
            # print(patch2)


            #Loading the image embedding
            try:
                image_embedding = self.image_embeddings[i]

            except:
                #Normalisation of the image.
                # intensity_transforms = tio.Compose([tio.Clamp(-1000,1000), tio.ZNormalization(masking_method=lambda x: x > 0)]) if self.dataset_info['dataset_modality'].title() == 'CT' else tio.Compose([tio.ZNormalization(masking_method=lambda x: x > 0)])
                # #Need to squeeze into a 4D image
                # im3d = intensity_transforms(patch1['image']['data'][0,:])
                # #Now unsqueeze for the img encoder.
                # im3d = im3d.unsqueeze(dim=0)

                #NOTE: We have, following the protocol used commonly in the SAM-Med3D implementation, decided to move the image normalization process into the original 
                #subject preparation function. 
                im3d = patch1['image']['data']

                #Patch size: 1 x 1 x self.patch_size (128 x 128 x 128)
                with torch.no_grad():
                    image_embedding = self.sam_model.image_encoder(
                        im3d.to(self.infer_device)
                    )  # (1, 384, 16, 16, 16)

                self.image_embeddings[i] = image_embedding 

            #Loading prev_mask
            try:
                low_res_masks = self.low_res_masks[i] 
            except: 
                low_res_masks = F.interpolate(
                    torch.zeros_like(im3d).to(self.infer_device).float(),
                    size=(self.patch_size[0] // 4, self.patch_size[1] // 4, self.patch_size[2] // 4),
                )

            with torch.no_grad():
                points_input = []
                labels_input = []
                #Extracting the prompt coordinates:
                for class_lb, class_code in self.dummy_config_labels.items():
                    original_code = self.config_labels[class_lb]
                    #SAMMed3D expects that prompts are provided in the following format: N x 1 x N_Dim meanwhile the labels are intended to be provided in the format: N x 1
                    coors = torch.argwhere(patch2['label']['data'][0,0,:] == class_code).unsqueeze(dim=1).to(device=self.infer_device, dtype=torch.int64)
                    if coors.numel():
                        coors_lbs = (torch.ones(coors.shape[0], 1) * original_code).to(device=self.infer_device, dtype=torch.int64)
                        points_input.append(coors)
                        labels_input.append(coors_lbs)
                    else:
                        continue 
                if not points_input:
                    input_points = None 
                else: 
                    points_input = torch.cat(points_input, dim=0)
                    labels_input = torch.cat(labels_input, dim=0)

                    print(f'points are {points_input}')
                    print(f'The patch location is {patch1["location"]}') 

                    #Note: SAM-Med3D appears to be incapable of handling multiple input prompt coordinates simultaneously, hence we will randomly sample one of the extracted
                    #points, as an approximation to the original point. NOTE: This is very suboptimal, and may lead to some real points being ignored as a result also.
                    #(i.e. not the "pseudo" points generated as a byproduct of resampling) if more than one point falls into the same patch.

                    random_point_n = torch.randint(low=0, high=points_input.shape[0], size=(1,))
                    points_input = points_input[random_point_n, :] 
                    labels_input = labels_input[random_point_n, :]

                    input_points = [points_input, labels_input]

                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=input_points,
                    boxes=None,
                    masks=low_res_masks.to(self.infer_device),
                )
                low_res_masks, _ = self.sam_model.mask_decoder(
                    image_embeddings=image_embedding.to(self.infer_device),  # (B, 384, 64, 64, 64)
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                    multimask_output=False,
                )
                upsampled_mask = F.interpolate(
                    low_res_masks,
                    size=patch1['image']['data'].shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

                #Saving the low res mask for the next iteration:
                self.low_res_masks[i] = low_res_masks 

                #Converting the output logits into probabilistic map. 

                seg_prob = torch.sigmoid(upsampled_mask)  # (B, 1, 64, 64, 64)
              
            aggregator.add_batch(seg_prob, locations=patch1['location'])
        
        return tio.Subject(image=tio.ScalarImage(tensor=aggregator.get_output_tensor(), affine=subject['image'].affine))
  
    def process_output(self, request_subj, input_subj, output_subj):
        #Returns probs_tensor: Channel split tensor divided in order of the class configs CHWD, pred (discretised) 1HWD, affine: affine output tensor.
        
        #We cannot use any invertible transforms as the resampling operation has no invertible equivalent in tio, but we need to undo 
        # to return the image in the original image domain unlike SAMMed3Ds implementation which presumes the input spacing is 1.5mm^3 always. 

        #Moreover, we must use the resizing transform which is extremely suboptimal, the resampling not being an invertible transform means that we cannot transfer the
        #trace info for undoing. Moreover, a naive resampling will not ensure that the output spatial dimensions match the inputs!
        
        # This is probably the simplest way to minimise information loss while ensuring spatial dimension consistency when passing it back to the original image domain. 
    
        resizer = tio.Resize(target_shape=request_subj['image'].data.shape[1:], image_interpolation='linear')     
        resized_output = resizer(output_subj)
                
        probs = copy.deepcopy(resized_output['image'].data)
        #Here we will perform a sleight of hand and just copy the original affine tensor. This is because the resizing operation will corrupt the output affine array. 
        affine = torch.from_numpy(copy.deepcopy(request_subj['image'].affine)).to(dtype=torch.int64)

        if probs.shape[0] == 1:
            #In the case that the probabilistic map is a single channel, then it represents the probabilistic map of the foreground only! 
            # The background will go into the 0th channel s.t. argmaxing will result in a 0 value.
            probs_maps = []
            for class_lb in self.config_labels.keys():
                if class_lb.title() == 'Background':
                    probs_maps.append(1 - probs)
                else:
                    probs_maps.append(probs) 
            probs_maps = torch.cat(probs_maps, dim=0) 
        
        
            pred = torch.argmax(probs_maps, dim=0).unsqueeze(dim=0).to(dtype=torch.int64) 
        else:
            NotImplementedError('Multiclass not yet supported.')

        return probs_maps, pred, affine 
    
    def prompt_prep(self, request, prompts_tensor, is_state):
    
        #Splitting prompts by class, and performing a re-mapping with the assumption that integer codes >= 0.
        for ptype, ps in is_state['interaction_dict_format'].items():
            if ps is not None:
                assert ptype.title() == 'Points'
            else:
                continue 

            #Splitting by class and mapping/projecting onto the prompts tensor.
            for class_lb in request['config_labels_dict'].keys():
                if ps[class_lb] == []:
                    continue 
                else: 
                    if ptype.title() == 'Points':
                        p_coords = torch.stack([torch.tensor(i) for i in ps[class_lb]], dim=0) #Here each list is nested once
                    elif ptype.title() == 'Scribbles':
                        p_coords = torch.cat([torch.tensor(i) for i in ps[class_lb]], dim=0) #Here each list is nested twice.

                    #We then inject the set of prompts into the prompts tensor according to the coords and dummy_config label.  
                    #First we append a 0 index as the spatial coordinates of the prompt do not reflect the channel-first nature of the image.
                    p_coords = torch.cat([torch.zeros(p_coords.shape[0],1), p_coords], dim=1)                      
                    prompts_tensor[tuple(p_coords.to(torch.int64).T)] = self.dummy_config_labels[class_lb]
    
        return prompts_tensor 
    
    
    def subject_prep(self, request):
        img = request['image']['metatensor']
        affine = copy.deepcopy(request['image']['metatensor'].meta['affine'])
        shape = img.shape 
    
        #Ordering the set of interaction states provided, first check if there is an initialisation: if so, place that first. 
        im_order = []
        
        init_modes  = {'Automatic Init', 'Interactive Init'}

        edit_names_list = list(set(request['im']).difference(init_modes))

        #Sorting this list.
        edit_names_list.sort(key=lambda test_str : list(map(int, re.findall(r'\d+', test_str))))

        #Extending the ordered list. 
        #
        im_order.extend(edit_names_list) 


        #Creating a dummy configs labels mapping such that we can modify the prompt label code for transformation into segmentation image-domain using image-level transforms.
        self.dummy_config_labels = {key:val+1 for key,val in request['config_labels_dict'].items()}
        #Also adding the config labels to the attributes of this inference app class.
        self.config_labels = request['config_labels_dict']



        #Creating the prompt object. 

        #Creating a prompts array, we will not include bbox as an accepted prompt i.e. we will discard this. We will treat points and scribbles identically.
        prompts_tensor = torch.zeros(img.shape).to(dtype=torch.int64)

        #Extracting the prompts according to the current mode of inference. 
        
        if request['model'] == 'IS_interactive_edit':
            #In this case we are working with an interactive edit:
            key = edit_names_list[-1]
            is_state = request['im'][key]

            prompts_tensor = self.prompt_prep(request=request,prompts_tensor=prompts_tensor, is_state=is_state)

            assert self.image_embeddings, 'Image embeddings must exist and be stored for editing'
            assert self.low_res_masks, 'Low res forward propagated masks must exist and be stored for editing'

        elif request['model'] == 'IS_interactive_init':
            key = 'Interactive Init' 
            #TODO: Add some wiping of variables/re-init.
            self.image_embeddings = {}
            self.low_res_masks = {} 

            is_state = request['im'][key]
            prompts_tensor = self.prompt_prep(request=request,prompts_tensor=prompts_tensor, is_state=is_state)

        elif request['model'] == 'IS_autoseg':
            key = 'Automatic Init'
            
            #TODO: Add some wiping of variables/re-init.
            self.image_embeddings = {}
            self.low_res_masks = {} 
        
        #Creating the image object
        img_obj = tio.ScalarImage(tensor=torch.from_numpy(copy.deepcopy(img.array)), affine=affine)
        #Creating the prompts obj (we treat it like a label map where coords are represented with delta functions)
        prompts_obj = tio.LabelMap(tensor=prompts_tensor, affine=affine)
        #Creating the subject
        subject = tio.Subject(image=img_obj, label=prompts_obj)
        #Orientation
        orientation = tio.ToCanonical() 
        #Create the resampler. 
        resampler = tio.Resample(target=self.target_spacing, image_interpolation='linear', label_interpolation='nearest')
        
        #Now we load-in according to the desired spacing for SAMMed3d.
        # resampled_subj = resampler(subject) 
        #Applying the transforms.
        
        prepped_subj = tio.Compose([orientation, resampler])(subject)

        return subject, prepped_subj
    
    def build_inference_apps(self):
        #Building the inference app, needs to have an end to end system in place for each "model" type which can be passed by the request: 
        # 
        # IS_autoseg, IS_interactive_init, IS_interactive_edit. (all are intuitive wrt what they represent.) 
        
        self.infer_apps = {
            'IS_autoseg':{'binary_pad':self.binary_pad_inference, 'binary_sw':self.binary_sw_inference},
            'IS_interactive_init': {'binary_pad':self.binary_pad_inference, 'binary_sw':self.binary_sw_inference},
            'IS_interactive_edit': {'binary_pad':self.binary_pad_inference, 'binary_sw':self.binary_sw_inference}
            }
    def app_configs(self):

        #STRONGLY Recommended: A method which returns any configuration specific information for printing to the logfile. Expects a dictionary format.
        return {'infer_app_name': self.app_params['checkpoint_name']} 
    
    def __call__(self, request: dict):
        
        '''
        Input request dictionary for application contains the following input fields:

        NOTE: All input arrays, tensors etc, will be on CPU. NOT GPU. 

        NOTE: Orientation convention is always assumed to be RAS! 

            image: A dictionary containing a path & a pre-loaded (UI) metatensor objects 
            {
            'metatensor':monai metatensor object containing image, torch.float datatype.
            'meta_dict': image_meta_dictionary, contains the original affine from loading and current affine array in the ui-domain.}

            model: A string denoting the inference "mode" being simulated, has three options: 
                    1) Automatic Segmentation, denoted: 'IS_autoseg' 
                    2) Interactive Initialisation: 'IS_interactive_init'
                    3) Interactive Editing: 'IS_interactive_edit'
            
            config_labels_dict: A dictionary containing the class label - class integer code mapping relationship being used, note that the codes are >= 0 with 0 = background.

            im: An interaction memory dictionary containing the set of interaction states. 
            Keys = Infer mode + optionally (for Edit) the inference iter num (1, ...).       

            Within each interaction state in IM:    
            
            prev_probs: A dictionary containing: {
                    'metatensor': Non-modified (CHWD) metatensor/torch tensor that is forward-propagated from the prior output (CHWD).
                    'meta_dict': Non-modified meta dictionary that is forward propagated.
                    }
            prev_pred: A dictionary containing: {
                    'metatensor': Non-modified metatensor/torch tensor that is forward-propagated from the prior output (1HWD).
                    'meta_dict': Non-modified meta dictionary that is forward propagated.
                    }

            prompt information: See `<https://github.com/IS_Validate/blob/main/src/data/interaction_state_construct.py>`

            interaction_torch_format: A prompt-type separated dictionary containing the prompt information in list[torch.tensor/metatensor] format 
                {'interactions':dict[prompt_type_str[list[torch.tensor/metatensor] OR NONE ]], 
                'interactions_labels':dict[prompt_type_str_labels [list[torch.tensor/metatensor] OR NONE]],
                }
            interaction_dict_format: A prompt-type separated dictionary containing the prompt info in class separated dict format
                (where each prompt spatial coord is represented as a sublist).  
                dict[prompt_type_str[class[list[list]] OR NONE]]

        '''
        #Callback which is required, and will be what is used to process the input requests for inference. E.g.: NOTE: SAMMed3D was trained in RAS so no need for rotation.
        
        request_subj, input_subj = self.subject_prep(request=request) 

        #Now, determine whether the image size is too large or not for the 128 cubed patch size the model was trained with.
        if all([i <= self.patch_size[j] for j,i in enumerate(input_subj['image'].shape[1:])]):
            #In this case, the resampled image was < 128 cubed in all dimensions so we will just use the padding  
            infer_mode = 'pad'
        else:
            #In this case, the resampled image was > 128 cubed in all dimensions so we must use sw-inference/patch based inference.
            infer_mode = 'sw'

        if len(request['config_labels_dict']) == 2:
            class_type = 'binary'
        elif len(request['config_labels_dict']) > 2:
            class_type = 'multi'
            raise NotImplementedError 
        else:
            raise Exception('Should not have received less than two class labels at minimum')
        
        app = self.infer_apps[request['model']][f'{class_type}_{infer_mode}']
    
        output_subj = app(input_subj)

        probs_tensor, pred, affine = self.process_output(request_subj, input_subj, output_subj)

        assert probs_tensor.shape[1:] == request_subj['image'].shape[1:]
        assert pred.shape[1:] == request_subj['image'].shape[1:] 
        assert torch.all(affine == request['image']['metatensor'].meta['affine'])
        output = {
            'probs':{
                'metatensor':probs_tensor.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
            'pred':{
                'metatensor':pred.to(device='cpu'),
                'meta_dict':{'affine': affine.to(device='cpu')}
            },
        }
        return output 


# if __name__ == '__main__':
#     infer_app = InferApp(
#         {'dataset_name':'BraTS2021',
#         'dataset_modality':'MRI'}, torch.device('cuda'))

#     infer_app.app_configs()

#     request = {
#         'image':{
#             'metatensor': MetaTensor(torch.randn((1,120,120,77)).abs(), affine=torch.tensor([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])),
#             'meta_dict':{'affine':torch.tensor([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])}
#         },
#         # 'model':'IS_interactive_edit',
#         'model': 'IS_interactive_init',
#         'config_labels_dict':{'background':0, 'tumor':1},
#         'im':
        
#         # {'Automatic Init': None}
#         {'Interactive Init':{
#             'interaction_torch_format': {'interactions': {'points': [torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]])], 'scribbles': None, 'bboxes': None}, 'interactions_labels': {'points': [torch.tensor([0]), torch.tensor([1])], 'scribbles': None, 'bboxes': None}},  
#             'interaction_dict_format': {
#             'points': {'background': [[40, 103, 43]],
#             # 'tumor': [[62, 62, 39]]
#             'tumor':[[30,30,15]]
#             },
#             'scribbles': None,
#             'bboxes': None
#             },
#             'prev_probs': {'metatensor': None, 'meta_dict': None}, 
#             'prev_pred': {'metatensor': None, 'meta_dict': None}}
#         },
#         # {'Interactive Edit Iter 1':
#         # {'interaction_torch_format': {'interactions': {'points': [torch.tensor([[40, 103, 43]]), torch.tensor([[62, 62, 39]])], 'scribbles': None, 'bboxes': None}, 'interactions_labels': {'points': [torch.tensor([0]), torch.tensor([1])], 'scribbles': None, 'bboxes': None}}, 
#         # 'interaction_dict_format': {
#         # 'points': {'background': [[40, 103, 43]],
#         # 'tumor': [[62, 62, 39]]
#         # },
#         # 'scribbles': None,
#         # 'bboxes': None
#         # },
#         # 'prev_probs': {'metatensor': torch.randn(2,120,120,77), 'meta_dict': {'affine':torch.tensor([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])}}, 
#         # 'prev_pred': {'metatensor': torch.randn(1,120,120,77).to(dtype=torch.int64), 'meta_dict': {'affine':torch.tensor([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1]])}}
#         # }

#         # }

#     }
#     infer_app(request)
#     print('halt')