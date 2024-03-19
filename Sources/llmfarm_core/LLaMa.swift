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
    public var batch: llama_batch?
    public var hardware_arch: String=""
    public var temporary_invalid_cchars: [CChar]  = []
    public var progressCallback: ((Float)  -> (Bool))? = nil
    
    public override func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default, params:gpt_context_params,
                                        model_load_progress_callback:((Float)  -> (Bool))?) throws -> Bool{
        var context_params = llama_context_default_params()
        var model_params = llama_model_default_params()
        context_params.n_ctx = UInt32(contextParams.context)
        context_params.seed = UInt32(contextParams.seed)
        context_params.n_threads = UInt32(contextParams.n_threads)
        context_params.logits_all = contextParams.logitsAll
        //        context_params.n_batch = contextParams.
        model_params.vocab_only = contextParams.vocabOnly
        model_params.use_mlock = contextParams.useMlock
        model_params.use_mmap = contextParams.useMMap
        //        A C function pointer can only be formed from a reference to a 'func' or a literal closure
        self.progressCallback = model_load_progress_callback
        self.retain_new_self_ptr()
        model_params.progress_callback = { progress,b in
            //                let LLaMa_obj = Unmanaged<LLaMa>.fromOpaque(LLaMa_obj_ptr!).takeRetainedValue()
            //                let LLaMa_ptr = Unmanaged<LLaMa>.fromOpaque(LLaMa_obj!).takeRetainedValue()
//            LLaMa_obj?.retain_new_self_ptr()
            if (LLaMa_obj?.progressCallback != nil){
                let res = LLaMa_obj?.progressCallback!(progress)
                return res ?? false
            }
            
            return true
        }

//        if contextParams.use_metal{
//            model_params.n_gpu_layers = 100
//        }else{
//            model_params.n_gpu_layers = 0
//        }
//        self.hardware_arch = Get_Machine_Hardware_Name()// Disable Metal on intel Mac
//        if self.hardware_arch=="x86_64"{
//            model_params.n_gpu_layers = 0
//        }
        model_params.n_gpu_layers = get_gpu_layers()
        
#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        
        if contextParams.lora_adapters.count>0{
            model_params.use_mmap = false
        }
        
        llama_backend_init(false)
        
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
        //        var tokens_tmp: [llama_token] = [Int32](repeating: 0, count: 100000)
        //        var tokens_count:Int = 0
        //        llama_load_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",tokens_tmp.mutPtr, 100000,&tokens_count)
        //        self.session_tokens.append(contentsOf: tokens_tmp[0..<tokens_count])
        //        try? llm_eval(inputBatch:self.session_tokens)
        //        llama_load_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
        if !load_clip_model(){
            return false
        }
        self.batch = llama_batch_init(sampleParams.n_batch, 0, 1)
        return true
    }
    
    public func load_clip_model() -> Bool{
        return true
    }
    
    private func retain_new_self_ptr(){
        LLaMa_obj = Unmanaged<LLaMa>.fromOpaque(Unmanaged.passRetained(self).toOpaque()).takeRetainedValue()
        //        LLaMa_obj_ptr = Unmanaged.passRetained(self).toOpaque()
        //        LLaMa_obj_ptr = UnsafeMutablePointer(OpaquePointer(bitPattern: Unmanaged.passUnretained(self)))
        // LLaMa_ptr = Unmanaged<LLaMa_MModal>.fromOpaque(LLaMaMM_obj_ptr!).takeRetainedValue()
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
        //        llama_save_state(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state_.bin")
        //        llama_save_session_file(self.context,"/Users/guinmoon/Library/Containers/com.guinmoon.LLMFarm/Data/Documents/models/dump_state.bin",self.session_tokens, self.session_tokens.count)       
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
    
//    public func llm_eval_old(inputBatch:[ModelToken]) throws -> Bool{
//        var mutable_inputBatch = inputBatch
//        if llama_eval(self.context, mutable_inputBatch.mutPtr, Int32(inputBatch.count), min(self.contextParams.context, self.nPast)) != 0 {
//            return false
//        }
//        return true
//    }
    
     public override func llm_eval(inputBatch:[ModelToken]) throws -> Bool{
         var mutable_inputBatch = inputBatch
         if llama_eval(self.context, mutable_inputBatch.mutPtr, Int32(inputBatch.count), min(self.contextParams.context, self.nPast)) != 0 {
             return false
         }
//        if self.nPast==0{
//            completion_init(tokens_list:inputBatch)
//        }else{
//            llama_batch_clear(&batch!)
//            for i1:Int32 in 0..<Int32(inputBatch.count) {
//                llama_batch_add(&batch!, inputBatch[Int(i1)], self.nPast+i1, [0], false)
//            }
//            batch!.logits[Int(batch!.n_tokens) - 1] = 1
////            llama_batch_add(&batch!, inputBatch[0], self.nPast, [0], true)
//
////            n_decode += 1
////            n_cur    += 1
//
//            if llama_decode(context, batch!) != 0 {
//                print("failed to evaluate llama!")
//            }
//        }
        return true
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

    func completion_init(tokens_list: [ModelToken]) {
//        print("attempting to complete \"\(text)\"")

        // tokens_list = tokenize(text: text, add_bos: true)
        temporary_invalid_cchars = []

//        let n_ctx = llama_n_ctx(context)
//        let n_kv_req = tokens_list.count + (Int(n_len) - tokens_list.count)
//
//        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")
//
//        if n_kv_req > n_ctx {
//            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
//        }

//        for id in tokens_list {
//            print(String(cString: token_to_piece(token: id) + [0]))
//        }

        llama_batch_clear(&batch!)

        for i1:Int32 in 0..<Int32(tokens_list.count) {
            llama_batch_add(&batch!, tokens_list[Int(i1)], i1, [0], false)
        }
        batch!.logits[Int(batch!.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch!) != 0 {
            print("llama_decode() failed")
        }

//        n_cur = batch.n_tokens
    }

    private func token_to_piece(token: Int32) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(model, token, result, 8)
        
        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(model, token, newResult, -nTokens)
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
    
    
    
    
    public override func llm_tokenize(_ input: String) -> [ModelToken] {
        if input.count == 0 {
            return []
        }
        
        //        llama_tokenize(
        //                struct llama_context * ctx,
        //                          const char * text,
        //                                 int   text_len,
        //                         llama_token * tokens,
        //                                 int   n_max_tokens,
        //                                bool   add_bos)
        let n_tokens = Int32(input.utf8.count) + (self.contextParams.add_bos_token == true ? 1 : 0)
        var embeddings: [llama_token] = Array<llama_token>(repeating: llama_token(), count: input.utf8.count)
        let n = llama_tokenize(self.model, input, Int32(input.utf8.count), &embeddings, n_tokens, self.contextParams.add_bos_token, self.contextParams.parse_special_tokens)
        if n<=0{
            return []
        }
        if Int(n) <= embeddings.count {
            embeddings.removeSubrange(Int(n)..<embeddings.count)
        }
        
        if self.contextParams.add_eos_token {
            embeddings.append(llama_token_eos(self.context))
        }
        
        return embeddings
    }
}

