//
//  LLaMa.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

// var LLaMa_obj_ptr:UnsafeMutableRawPointer? = nil
var LLaMa_obj:LLaMa? = nil
//var LLaMa_ptr:UnsafeMutablePointer? = nil

public class LLaMa: LLMBase {
    
    public var model: OpaquePointer?
    public var ctx_sampling: OpaquePointer?
    public var batch: llama_batch?
    public var hardware_arch: String=""
    public var temporary_invalid_cchars: [CChar]  = []    

    public func init_sampling_param(){
        self.ctx_sampling = init_sampling(sampleParams.repeat_last_n,
                                            sampleParams.top_k,
                                            sampleParams.top_p,
                                            sampleParams.min_p,
                                            sampleParams.tfs_z,
                                            sampleParams.typical_p,
                                            sampleParams.temp,
                                            0.0,
                                            1.0,
                                            sampleParams.repeat_last_n,
                                            sampleParams.repeat_penalty,
                                            sampleParams.frequence_penalty,
                                            sampleParams.presence_penalty,
                                            sampleParams.mirostat,
                                            sampleParams.mirostat_tau,
                                            sampleParams.mirostat_eta,
                                            sampleParams.penalize_nl,
                                            0 /*SEED*/,
                                            self.contextParams.grammar_path ?? "");
    }

    public override func llm_load_model(path: String = "", 
                                        contextParams: ModelAndContextParams = .default,
                                        params:gpt_context_params) throws -> Bool {
        var context_params = llama_context_default_params()
        var model_params = llama_model_default_params()
        init_sampling_param()
                
        context_params.n_ctx = UInt32(contextParams.context)
        context_params.seed = UInt32(contextParams.seed)
        context_params.n_threads = UInt32(contextParams.n_threads)
        context_params.logits_all = contextParams.logitsAll
//        context_params.flash_attn = contextParams.flash_attn        
        context_params.flash_attn = false

        model_params.vocab_only = contextParams.vocabOnly
        model_params.use_mlock = contextParams.useMlock
        model_params.use_mmap = contextParams.useMMap        

        self.retain_new_self_ptr()
        model_params.progress_callback = { progress,b in
            if (LLaMa_obj?.modelLoadProgressCallback != nil){
                let res = LLaMa_obj?.modelLoadProgressCallback!(progress)
                return res ?? false
            }
            return true
        }

        // context_params.cb_eval = { t, ask, user_data in
        //     //    var  t:ggml_tensor? = a?.pointee
        //     //    let t_name = String(cString:get_tensor_name(t))            
        //     if (LLaMa_obj?.evalCallback != nil){
        //         //    let res = LLaMa_obj?.evalCallback!( Int(check_tensor_name(t)))
        //         _ = LLaMa_obj?.evalCallback!( 0 )
        //         return false
        //     }
        //     return false
        // };
       
        /*
        if contextParams.use_metal{
            model_params.n_gpu_layers = 100
        }else{
            model_params.n_gpu_layers = 0
        }
        self.hardware_arch = Get_Machine_Hardware_Name()// Disable Metal on intel Mac
        if self.hardware_arch=="x86_64"{
            model_params.n_gpu_layers = 0
        }
         */
        model_params.n_gpu_layers = get_gpu_layers()
        
#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        
        if contextParams.lora_adapters.count>0{
            model_params.use_mmap = false
        }
        _ = self.modelLoadProgressCallback?(0)        
        llama_backend_init()
        self.model = llama_load_model_from_file(path, model_params)
        if self.model == nil{
            return false
        }

        for lora in contextParams.lora_adapters{
            llama_model_apply_lora_from_file(model,lora.0,lora.1,nil,6);
        }
        self.context = llama_new_context_with_model(self.model, context_params)
        if self.context == nil {
            return false
        }
        
        if !load_clip_model(){
            return false
        }
        
        self.load_state()
        
        self.batch = llama_batch_init(sampleParams.n_batch, 0, 1)
        return true
    }
    

    public override func llm_sample() -> ModelToken {
        let id  = spm_llama_sampling_sample(self.ctx_sampling, self.context, nil,-1);
        spm_llama_sampling_accept(self.ctx_sampling, self.context,  id, /* apply_grammar= */ true);
        return id;
    }
    
    public override func load_state(){
        if self.contextParams.save_load_state &&
            self.contextParams.state_dump_path != "" &&
            FileManager.default.fileExists(atPath: self.contextParams.state_dump_path)
        {
                var tokens_tmp: [llama_token] = [Int32](repeating: 0, count: 4096)
                var tokens_count:Int = 0
                llama_state_load_file(self.context,self.contextParams.state_dump_path,tokens_tmp.mutPtr, 4096,&tokens_count)
                if (tokens_count>0){
                    self.outputRepeatTokens.append(contentsOf: tokens_tmp[0..<tokens_count-1])
                    self.nPast = tokens_tmp[tokens_count-1]
                }                
        }
    }
    
    public override func save_state(){
        if self.contextParams.save_load_state &&
            self.contextParams.state_dump_path != "" {
            self.outputRepeatTokens.append(self.nPast)
            llama_state_save_file(self.context,self.contextParams.state_dump_path,self.outputRepeatTokens, self.outputRepeatTokens.count)
        }
    }
    
    private func retain_new_self_ptr(){
        LLaMa_obj = Unmanaged<LLaMa>.fromOpaque(Unmanaged.passRetained(self).toOpaque()).takeRetainedValue()
    }
    
    public override func destroy_objects(){
        print("destroy LLaMa")
        if batch != nil{
            llama_batch_free(batch!)
        }
        if context != nil{
            llama_free(context)
        }
        if model != nil{
            llama_free_model(model)
        }
        self.destroy_clip()
//        llama_backend_free()
    }
    
    public func destroy_clip(){
        
    }
    
    deinit {
        self.save_state()
        print("deinit LLaMa")
        self.destroy_objects()
        print("LLaMa deinited")
    }
    
    override func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32{
        return Int32(llama_n_ctx(self.context))
    }
    
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return llama_n_vocab(self.model)
    }
    
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return llama_get_logits(self.context);
    }
    
    public override func load_grammar(_ path:String) throws -> Void{ }
    
    public override func llm_eval(inputBatch: inout [ModelToken]) throws -> Bool {
        if llama_decode(context,llama_batch_get_one(&inputBatch, Int32(inputBatch.count), self.nPast, 0)) != 0 {
            print("failed to evaluate llama!")
            return false
        }
        return true
    }

    // func llm_eval_ex(inputBatch: inout [ModelToken]) throws -> Bool {
    //         if self.nPast==0{
    //         completion_init(tokens_list:inputBatch)
    //     }else{
    //         llama_batch_clear(&batch!)
    //         for i1 in 0..<inputBatch.count {
    //             let i = Int(i1)
    //             llama_batch_add(&batch!, inputBatch[i], Int32(i)+self.nPast, [0], true)
    //         }
    //         if llama_decode(context, batch!) != 0 {
    //             print("failed to evaluate llama!")
    //             return false
    //         }
    //     }   
    //     return true
    // } 

    // func completion_init(tokens_list: [ModelToken]) {
    //     temporary_invalid_cchars = []
    //     llama_batch_clear(&batch!)
    //     for i1 in 0..<tokens_list.count {
    //         let i = Int(i1)
    //         llama_batch_add(&batch!, tokens_list[i], Int32(i), [0], false)
    //     }
    //     batch!.logits[Int(batch!.n_tokens) - 1] = 1
    //     if llama_decode(context, batch!) != 0 {
    //         print("llama_decode() failed")
    //     }
    // }

    public override func kv_shift() throws{
        let n_discard = self.nPast/2
        llama_kv_cache_seq_rm (context, 0, 0            , n_discard);
        llama_kv_cache_seq_add(context, 0, n_discard, self.nPast, -n_discard);      
        self.nPast -= n_discard;
        try ExceptionCather.catchException {
            var in_batch = [self.llm_token_eos()]
            _ = try? self.llm_eval(inputBatch: &in_batch)
        }
        self.nPast+=1
        self.outputRepeatTokens = []
        print("Context Limit!")
    }
    

    override func llm_init_logits() throws -> Bool {
        return true
    }
    
    func llama_batch_clear(_ batch: inout llama_batch) {
        batch.n_tokens = 0
    }

    func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
        batch.token   [Int(batch.n_tokens)] = id
        batch.pos     [Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

        batch.n_tokens += 1
    }
    
    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    

    private func token_to_piece(token: Int32) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
//        llama_token_to_piece(const struct llama_model * model, llama_token token, char * buf, int32_t length, bool special)
        let nTokens = llama_token_to_piece(model, token, result, 8,/*true*/self.contextParams.parse_special_tokens)
        
        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(model, token, newResult, -nTokens,/*true*/self.contextParams.parse_special_tokens)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
    
    public override func llm_token_to_str(outputToken:Int32) -> String? {
        //        if let cStr = llama_token_to_str(context, outputToken){
        //            return String(cString: cStr)
        //        }
        //        return nil
        let new_token_cchars = token_to_piece(token: outputToken)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        return new_token_str
    }
    
    public override func llm_token_nl() -> ModelToken{
        return llama_token_nl(self.model)
    }
    
    public override func llm_token_bos() -> ModelToken{
        return llama_token_bos(self.model)
    }
    
    public override func llm_token_eos() -> ModelToken{
        return llama_token_eos(self.model)
    }

    public override func llm_tokenize(_ input: String, add_bos: Bool?, parse_special:Bool?) -> [ModelToken] {
        if input.count == 0 {
            return []
        }
        let n_tokens = Int32(input.utf8.count) + (self.contextParams.add_bos_token == true ? 1 : 0)
        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(self.model, input, Int32(input.utf8.count), &embeddings, n_tokens, 
                               add_bos ?? self.contextParams.add_bos_token,
                               parse_special ?? self.contextParams.parse_special_tokens)
        if n<=0{
            return []
        }
        if Int(n) <= embeddings.count {
            embeddings.removeSubrange(Int(n)..<embeddings.count)
        }
        
        if self.contextParams.add_eos_token {
            embeddings.append(llm_token_eos())
        }
        
        return embeddings
    }
}

