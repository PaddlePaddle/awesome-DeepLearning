self.model = build_model(cfg.model)
self.lr_schedulers = self.model.setup_lr_schedulers(cfg.lr_scheduler)  
self.optimizers = self.model.setup_optimizers(self.lr_schedulers,cfg.optimizer)
def visual(self,
               results_dir,
               visual_results=None,
               step=None,
               is_save_image=False):
        """
        visual the images, use visualdl or directly write to the directory

        Parameters:
            results_dir (str)     --  directory name which contains saved images
            visual_results (dict) --  the results images dict
            step (int)            --  global steps, used in visualdl
            is_save_image (bool)  --  weather write to the directory or visualdl
        """
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        min_max = self.cfg.get('min_max', None)
        if min_max is None:
            min_max = (-1., 1.)

        image_num = self.cfg.get('image_num', None)
        if (image_num is None) or (not self.enable_visualdl):
            image_num = 1
        for label, image in visual_results.items():
            image_numpy = tensor2img(image, min_max, image_num)
            if (not is_save_image) and self.enable_visualdl:
                self.vdl_logger.add_image(
                    results_dir + '/' + label,
                    image_numpy,
                    step=step if step else self.global_steps,
                    dataformats="HWC" if image_num == 1 else "NCHW")
            else:
                if self.cfg.is_train:
                    msg = 'epoch%.3d_' % self.current_epoch
                else:
                    msg = ''
                makedirs(os.path.join(self.output_dir, results_dir))
                img_path = os.path.join(self.output_dir, results_dir,
                                        msg + '%s.png' % (label))
                save_image(image_numpy, img_path)
